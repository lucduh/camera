import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from donut_camera.data import Sample, load_samples, parse_output, values_by_field
from donut_camera.metrics import extraction_metrics
from donut_camera.model import DonutBundle
from donut_camera.timing import synchronize


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


def evaluate(
    bundle: DonutBundle,
    data_json: str | Path,
    *,
    batch_size: int,
    workers: int,
    max_new_tokens: int,
    debug_output: str | Path | None = None,
) -> dict:
    samples = load_samples(data_json)
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
            text = bundle.processor.tokenizer.decode(ids, skip_special_tokens=False)
            parsed = parse_output(text, bundle.task)
            reference = {
                field: value
                for field, value in values_by_field(
                    sample, duplicate_policy=bundle.task.duplicate_policy
                ).items()
                if field in bundle.task.fields
            }
            records.append(
                {
                    "prediction": parsed.fields,
                    "reference": reference,
                    "valid": parsed.valid,
                }
            )
            per_document.append(
                {
                    "document": anonymous_id(bundle.task.name, sample.document_id),
                    "generated_tokens": generated_length(
                        ids,
                        prompt_length=prompt.shape[1],
                        eos=bundle.processor.tokenizer.eos_token_id,
                        pad=bundle.processor.tokenizer.pad_token_id,
                    ),
                    "valid": parsed.valid,
                }
            )
    synchronize()
    wall_seconds = time.perf_counter() - started

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
