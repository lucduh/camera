import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from donut_camera.tasks import TaskSpec


@dataclass(frozen=True)
class Sample:
    document_id: str
    image: Path
    fields: tuple[dict[str, Any], ...]


def field_leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def load_samples(path: str | Path) -> list[Sample]:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if isinstance(raw, dict):
        raw = raw.get("samples", raw.get("records"))
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a JSON list of samples")

    samples = []
    for index, record in enumerate(raw):
        image = Path(record["image"])
        if not image.is_absolute():
            image = path.parent / image
        document_id = str(record.get("id", index))
        samples.append(
            Sample(
                document_id=document_id,
                image=image,
                fields=tuple(record.get("fields", ())),
            )
        )
    return samples


def values_by_field(sample: Sample) -> dict[str, str]:
    values: dict[str, str] = {}
    for annotation in sample.fields:
        field = field_leaf(str(annotation["field_name"]))
        if field not in values:
            values[field] = str(annotation.get("annotator_text", "")).strip()
    return values


def format_target(sample: Sample, task: TaskSpec) -> str:
    values = values_by_field(sample)
    parts = []
    for field in task.fields:
        value = values.get(field, "") or task.missing_token
        parts.extend((task.open_token(field), value, task.close_token(field)))
    return "".join(parts)


@dataclass(frozen=True)
class ParsedOutput:
    fields: dict[str, str]
    valid: bool


def parse_output(text: str, task: TaskSpec) -> ParsedOutput:
    fields: dict[str, str] = {}
    cursor = 0
    valid = True
    for field in task.fields:
        opening = task.open_token(field)
        closing = task.close_token(field)
        start = text.find(opening, cursor)
        end = text.find(closing, start + len(opening)) if start >= 0 else -1
        if start < cursor or end < 0:
            valid = False
            continue
        value = text[start + len(opening) : end].strip()
        fields[field] = "" if value == task.missing_token else value
        cursor = end + len(closing)

    for field in task.fields:
        if text.count(task.open_token(field)) != 1:
            valid = False
        if text.count(task.close_token(field)) != 1:
            valid = False
    return ParsedOutput(fields=fields, valid=valid)


class DonutDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        processor,
        task: TaskSpec,
        max_length: int,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.task = task
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        pixels = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        target = format_target(sample, self.task) + self.processor.tokenizer.eos_token
        encoded = self.processor.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels = encoded.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixels, "labels": labels}


def sanitize_text(text: str) -> str:
    """Remove control characters before serializing optional debug text."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
