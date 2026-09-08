import hashlib
import json
import time
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from donut_camera.data import Sample, load_samples, parse_output, values_by_field
from donut_camera.metrics import extraction_metrics
from donut_camera.model import DonutBundle
from donut_camera.timing import summarize_ms, synchronize, timed


class EvaluationDataset(Dataset):
    def __init__(self, samples: list[Sample], processor) -> None:
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        pixels = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        return pixels, sample


def collate_evaluation(batch):
    pixels, samples = zip(*batch)
    return torch.stack(pixels), list(samples)


def anonymous_id(task: str, document_id: str) -> str:
    return hashlib.sha256(f"{task}:{document_id}".encode()).hexdigest()[:16]


def generated_length(
    ids: torch.Tensor, *, prompt_length: int, eos: int, pad: int
) -> int:
    tokens = ids[prompt_length:].tolist()
    length = 0
    for token in tokens:
        if token == pad:
            break
        length += 1
        if token == eos:
            break
    return length


def transfer_pixels(cpu_pixels, *, device, dtype):
    return cpu_pixels.to(device, dtype=dtype, non_blocking=False)


def encode_pixels(model, pixels):
    return model.encoder(pixels, return_dict=True)


def decode_from_encoder(model, pixels, encoder_outputs, prompt, *, max_new_tokens: int):
    return model.generate(
        pixel_values=pixels,
        encoder_outputs=encoder_outputs,
        decoder_input_ids=prompt,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )


def parse_document(bundle: DonutBundle, ids: torch.Tensor, sample: Sample) -> dict:
    text = bundle.processor.tokenizer.decode(ids, skip_special_tokens=False)
    parsed = parse_output(text, bundle.task)
    reference = {
        field: value
        for field, value in values_by_field(
            sample, duplicate_policy=bundle.task.duplicate_policy
        ).items()
        if field in bundle.task.fields
    }
    return {
        "prediction": parsed.fields,
        "reference": reference,
        "valid": parsed.valid,
    }


def finish_evaluation(
    bundle: DonutBundle,
    records: list[dict],
    per_document: list[dict],
    wall_seconds: float,
    debug_output: str | Path | None,
) -> dict:
    if debug_output is not None:
        debug_path = Path(debug_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    return {
        "timing": {
            "documents": len(records),
            "wall_seconds": wall_seconds,
            "documents_per_second": len(records) / wall_seconds
            if wall_seconds
            else 0.0,
        },
        "strict": extraction_metrics(records, bundle.task),
        "normalized": extraction_metrics(records, bundle.task, normalized=True),
        "per_document": per_document,
    }


def evaluate_profiled(
    bundle: DonutBundle,
    samples: list[Sample],
    *,
    max_new_tokens: int,
    debug_output: str | Path | None,
) -> dict:
    """Profile the complete batch-one path without DataLoader prefetching."""
    model = bundle.model.eval()
    records = []
    per_document = []
    synchronize()
    overall_started = time.perf_counter()

    for sample in tqdm(samples, desc="evaluate and profile"):
        total_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        cpu_pixels = bundle.processor(
            image, return_tensors="pt"
        ).pixel_values.contiguous()
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1_000

        pixels, transfer_ms = timed(
            partial(
                transfer_pixels,
                cpu_pixels,
                device=bundle.device,
                dtype=bundle.dtype,
            )
        )
        prompt = bundle.decoder_start_ids(1)
        if bundle.device.type == "cuda":
            start_allocated = torch.cuda.memory_allocated(bundle.device)
            torch.cuda.reset_peak_memory_stats(bundle.device)
        else:
            start_allocated = None

        with torch.inference_mode():
            encoder_outputs, encoder_ms = timed(partial(encode_pixels, model, pixels))
            decode = partial(
                decode_from_encoder,
                model,
                pixels,
                encoder_outputs,
                prompt,
                max_new_tokens=max_new_tokens,
            )
            sequences, decoder_ms = timed(decode)

        parsing_started = time.perf_counter()
        record = parse_document(bundle, sequences[0], sample)
        parsing_ms = (time.perf_counter() - parsing_started) * 1_000
        total_ms = (time.perf_counter() - total_started) * 1_000
        records.append(record)

        if bundle.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(bundle.device)
            memory = {
                "start_allocated_mb": start_allocated / 1024**2,
                "peak_allocated_mb": peak_allocated / 1024**2,
                "incremental_peak_mb": (peak_allocated - start_allocated) / 1024**2,
            }
        else:
            memory = None
        per_document.append(
            {
                "document": anonymous_id(bundle.task.name, sample.document_id),
                "generated_tokens": generated_length(
                    sequences[0],
                    prompt_length=prompt.shape[1],
                    eos=bundle.processor.tokenizer.eos_token_id,
                    pad=bundle.processor.tokenizer.pad_token_id,
                ),
                "valid": record["valid"],
                "preprocessing_ms": preprocessing_ms,
                "transfer_ms": transfer_ms,
                "encoder_ms": encoder_ms,
                "decoder_ms": decoder_ms,
                "parsing_ms": parsing_ms,
                "total_ms": total_ms,
                "memory": memory,
            }
        )

    wall_seconds = time.perf_counter() - overall_started
    result = finish_evaluation(
        bundle, records, per_document, wall_seconds, debug_output
    )
    phases = (
        "preprocessing_ms",
        "transfer_ms",
        "encoder_ms",
        "decoder_ms",
        "parsing_ms",
        "total_ms",
    )
    result["profile"] = {
        phase: {
            "raw_ms": [row[phase] for row in per_document],
            **summarize_ms([row[phase] for row in per_document]),
        }
        for phase in phases
    }
    return result


def evaluate_batched(
    bundle: DonutBundle,
    samples: list[Sample],
    *,
    batch_size: int,
    workers: int,
    max_new_tokens: int,
    debug_output: str | Path | None,
) -> dict:
    loader = DataLoader(
        EvaluationDataset(samples, bundle.processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=bundle.device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_evaluation,
    )
    bundle.model.eval()
    records = []
    per_document = []
    synchronize()
    started = time.perf_counter()
    for pixels, batch_samples in tqdm(loader, desc="evaluate"):
        pixels = pixels.to(bundle.device, dtype=bundle.dtype, non_blocking=True)
        prompt = bundle.decoder_start_ids(len(batch_samples))
        with torch.inference_mode():
            sequences = bundle.model.generate(
                pixel_values=pixels,
                decoder_input_ids=prompt,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        for ids, sample in zip(sequences, batch_samples, strict=True):
            record = parse_document(bundle, ids, sample)
            records.append(record)
            per_document.append(
                {
                    "document": anonymous_id(bundle.task.name, sample.document_id),
                    "generated_tokens": generated_length(
                        ids,
                        prompt_length=prompt.shape[1],
                        eos=bundle.processor.tokenizer.eos_token_id,
                        pad=bundle.processor.tokenizer.pad_token_id,
                    ),
                    "valid": record["valid"],
                }
            )
    synchronize()
    return finish_evaluation(
        bundle,
        records,
        per_document,
        time.perf_counter() - started,
        debug_output,
    )


def evaluate(
    bundle: DonutBundle,
    data_json: str | Path,
    *,
    batch_size: int,
    workers: int,
    max_new_tokens: int,
    limit: int | None = None,
    profile: bool = False,
    debug_output: str | Path | None = None,
) -> dict:
    samples = load_samples(data_json)
    if limit is not None:
        samples = samples[:limit]
    if profile:
        if batch_size != 1 or workers != 0:
            raise ValueError("Profile mode requires batch_size=1 and workers=0")
        return evaluate_profiled(
            bundle,
            samples,
            max_new_tokens=max_new_tokens,
            debug_output=debug_output,
        )
    return evaluate_batched(
        bundle,
        samples,
        batch_size=batch_size,
        workers=workers,
        max_new_tokens=max_new_tokens,
        debug_output=debug_output,
    )
