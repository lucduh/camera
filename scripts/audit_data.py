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
    duplicate_groups: Counter[str] = Counter()
    duplicate_extra_annotations: Counter[str] = Counter()
    identical_duplicate_groups: Counter[str] = Counter()
    conflicting_duplicate_groups: Counter[str] = Counter()
    maximum_occurrences: Counter[str] = Counter()
    unknown_fields: Counter[str] = Counter()
    token_lengths = []
    widths = []
    heights = []
    unreadable_images = 0

    for sample in samples:
        names = [field_leaf(str(field["field_name"])) for field in sample.fields]
        values_by_name: dict[str, list[str]] = {}
        for annotation, name in zip(sample.fields, names, strict=True):
            value = str(annotation.get("annotator_text", "")).strip()
            values_by_name.setdefault(name, []).append(value)
            field_counts[name] += 1
            if not value:
                empty_counts[name] += 1
            if name not in task.fields:
                unknown_fields[name] += 1

        duplicated = False
        for name, values in values_by_name.items():
            maximum_occurrences[name] = max(maximum_occurrences[name], len(values))
            if len(values) < 2:
                continue
            duplicated = True
            duplicate_groups[name] += 1
            duplicate_extra_annotations[name] += len(values) - 1
            if len(set(values)) == 1:
                identical_duplicate_groups[name] += 1
            else:
                conflicting_duplicate_groups[name] += 1
        duplicate_documents += int(duplicated)
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
        "duplicates": {
            "documents": duplicate_documents,
            "groups_by_field": dict(sorted(duplicate_groups.items())),
            "extra_annotations_by_field": dict(
                sorted(duplicate_extra_annotations.items())
            ),
            "identical_groups_by_field": dict(
                sorted(identical_duplicate_groups.items())
            ),
            "conflicting_groups_by_field": dict(
                sorted(conflicting_duplicate_groups.items())
            ),
            "maximum_occurrences_by_field": dict(sorted(maximum_occurrences.items())),
        },
        "unreadable_images": unreadable_images,
        "image_width": distribution(widths),
        "image_height": distribution(heights),
        "target_tokens": distribution(token_lengths),
        "truncation": {
            str(limit): sum(length > limit for length in token_lengths)
            for limit in (64, 80, 96, 128, 256, 512)
        },
    }
    path = write_record(
        args.output,
        config={
            "task": task.name,
            "data_file": args.data.name,
            "processor": args.processor,
            "duplicate_policy": task.duplicate_policy,
            "images_checked": not args.skip_images,
        },
        measurements=measurements,
    )
    print(path)
    print(measurements)


if __name__ == "__main__":
    main()
