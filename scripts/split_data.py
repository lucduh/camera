import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path


def rewritten_record(record: dict, *, source_dir: Path, output_dir: Path) -> dict:
    record = copy.deepcopy(record)
    image = Path(record["image"])
    if not image.is_absolute():
        image = (source_dir / image).resolve()
        record["image"] = os.path.relpath(image, output_dir.resolve())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a persistent train/validation split"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one")
    resolved_outputs = {
        args.train_output.resolve(),
        args.validation_output.resolve(),
        args.manifest.resolve(),
    }
    if args.data.resolve() in resolved_outputs:
        raise ValueError("Refusing to overwrite the source dataset")

    source_bytes = args.data.read_bytes()
    records = json.loads(source_bytes)
    if not isinstance(records, list):
        raise TypeError("The source JSON must contain a list")

    indices = list(range(len(records)))
    random.Random(args.seed).shuffle(indices)
    validation_count = round(len(records) * args.validation_fraction)
    validation_indices = set(indices[:validation_count])
    train_indices = [
        index for index in range(len(records)) if index not in validation_indices
    ]
    validation_indices_ordered = sorted(validation_indices)

    def write_subset(path: Path, selected: list[int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        subset = [
            rewritten_record(
                records[index], source_dir=args.data.parent, output_dir=path.parent
            )
            for index in selected
        ]
        path.write_text(json.dumps(subset, indent=2, ensure_ascii=False))

    write_subset(args.train_output, train_indices)
    write_subset(args.validation_output, validation_indices_ordered)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "source_file": args.data.name,
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "seed": args.seed,
                "validation_fraction": args.validation_fraction,
                "source_documents": len(records),
                "train_documents": len(train_indices),
                "validation_documents": len(validation_indices_ordered),
                "validation_source_indices": validation_indices_ordered,
            },
            indent=2,
        )
    )
    print(
        f"Created {len(train_indices)} training and "
        f"{len(validation_indices_ordered)} validation documents"
    )


if __name__ == "__main__":
    main()
