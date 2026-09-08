import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from donut_camera.tasks import TaskSpec

DEFAULT_MODEL = "naver-clova-ix/donut-base"
DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass
class DonutBundle:
    model: VisionEncoderDecoderModel
    processor: DonutProcessor
    task: TaskSpec

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def decoder_start_ids(self, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

    def set_resolution(self, height: int, width: int) -> None:
        set_resolution(self.model, self.processor, height, width)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
        (directory / "task.json").write_text(
            json.dumps(asdict(self.task), indent=2, ensure_ascii=False)
        )


def resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def register_task_tokens(processor: DonutProcessor, task: TaskSpec) -> None:
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": list(task.structural_tokens)}
    )


def configure_model(model, processor: DonutProcessor, task: TaskSpec) -> None:
    embeddings = model.decoder.get_input_embeddings()
    if embeddings.num_embeddings != len(processor.tokenizer):
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
    task_id = processor.tokenizer.convert_tokens_to_ids(task.task_token)
    if task_id == processor.tokenizer.unk_token_id:
        raise ValueError(f"Task token {task.task_token!r} is not registered")
    # Chapter 4 uses an explicit eager decoder baseline. Optimized decoder
    # implementations are introduced as controlled factors in Chapter 5.
    model.decoder.config._attn_implementation = "eager"
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = task_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.decoder_start_token_id = task_id
    model.generation_config.max_length = None


def load_bundle(
    source: str = DEFAULT_MODEL,
    *,
    task: TaskSpec,
    device: str | None = None,
    dtype: str = "bf16",
    training: bool = False,
) -> DonutBundle:
    device = resolve_device(device)
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype {dtype!r}; choose from {tuple(DTYPES)}")
    model_dtype = torch.float32 if training else DTYPES[dtype]
    processor = DonutProcessor.from_pretrained(source)
    register_task_tokens(processor, task)
    model = VisionEncoderDecoderModel.from_pretrained(source, dtype=model_dtype)
    configure_model(model, processor, task)
    model.to(device)
    model.train(training)
    return DonutBundle(model=model, processor=processor, task=task)


def set_resolution(model, processor: DonutProcessor, height: int, width: int) -> None:
    if height % 40 or width % 40:
        raise ValueError("Donut resolutions must be divisible by 40")
    model.encoder.config.image_size = [height, width]
    model.config.encoder.image_size = [height, width]
    processor.image_processor.size = {"height": height, "width": width}


def autocast(device: str | torch.device, precision: str):
    device_type = torch.device(device).type
    enabled = device_type == "cuda" and precision == "bf16"
    return torch.autocast(
        device_type=device_type, dtype=torch.bfloat16, enabled=enabled
    )
