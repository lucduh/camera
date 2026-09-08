import argparse
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image
from transformers import DonutProcessor

from donut_camera.data import field_leaf, format_target, load_samples
from donut_camera.model import DEFAULT_MODEL, register_task_tokens
from donut_camera.records import write_record
from donut_camera.tasks import load_task


def distribution(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a Donut dataset without training"
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", default=DEFAULT_MODEL)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    task = load_task(args.task)
    samples = load_samples(args.data)
    processor = DonutProcessor.from_pretrained(args.processor)
    register_task_tokens(processor, task)

    field_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    duplicate_documents = 0
    unknown_fields: Counter[str] = Counter()
    token_lengths = []
    widths = []
    heights = []
    unreadable_images = 0

    for sample in samples:
        names = [field_leaf(str(field["field_name"])) for field in sample.fields]
        duplicate_documents += int(len(names) != len(set(names)))
        for annotation, name in zip(sample.fields, names, strict=True):
            field_counts[name] += 1
            if not str(annotation.get("annotator_text", "")).strip():
                empty_counts[name] += 1
            if name not in task.fields:
                unknown_fields[name] += 1
        target = format_target(sample, task) + processor.tokenizer.eos_token
        token_lengths.append(
            len(processor.tokenizer(target, add_special_tokens=False).input_ids)
        )
        if not args.skip_images:
            try:
                with Image.open(sample.image) as image:
                    width, height = image.size
                widths.append(width)
                heights.append(height)
            except OSError:
                unreadable_images += 1

    measurements = {
        "documents": len(samples),
        "configured_fields": list(task.fields),
        "field_counts": dict(sorted(field_counts.items())),
        "empty_counts": dict(sorted(empty_counts.items())),
        "unknown_fields": dict(sorted(unknown_fields.items())),
        "documents_with_duplicate_fields": duplicate_documents,
        "unreadable_images": unreadable_images,
        "image_width": distribution(widths),
        "image_height": distribution(heights),
        "target_tokens": distribution(token_lengths),
        "truncation": {
            str(limit): sum(length > limit for length in token_lengths)
            for limit in (64, 128, 256, 512)
        },
    }
    path = write_record(
        args.output,
        config={
            "task": task.name,
            "data_file": args.data.name,
            "processor": args.processor,
            "images_checked": not args.skip_images,
        },
        measurements=measurements,
    )
    print(path)
    print(measurements)


if __name__ == "__main__":
    main()
