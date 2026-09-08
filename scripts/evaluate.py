import argparse
from pathlib import Path

from donut_camera.evaluation import evaluate
from donut_camera.model import load_bundle, resolve_device
from donut_camera.records import write_record
from donut_camera.tasks import load_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Donut checkpoint"
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Evaluate only a prefix")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record batch-one pipeline timings; requires --batch-size 1 --workers 0",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Defaults to max_new_tokens from the task configuration",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    args = parser.parse_args()

    task = load_task(args.task)
    device = resolve_device(args.device)
    bundle = load_bundle(
        args.checkpoint,
        task=task,
        device=device,
        dtype=args.dtype,
    )
    bundle.set_resolution(args.height, args.width)
    max_new_tokens = args.max_new_tokens or task.max_new_tokens
    measurements = evaluate(
        bundle,
        args.data,
        batch_size=args.batch_size,
        workers=args.workers,
        max_new_tokens=max_new_tokens,
        limit=args.limit,
        profile=args.profile,
        debug_output=args.debug_output,
    )
    path = write_record(
        args.output,
        config={
            "task": task.name,
            "checkpoint": Path(args.checkpoint).name,
            "data_file": args.data.name,
            "height": args.height,
            "width": args.width,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "limit": args.limit,
            "profile": args.profile,
            "max_new_tokens": max_new_tokens,
            "dtype": args.dtype,
            "device": device,
        },
        measurements=measurements,
    )
    print(path)
    print(measurements["strict"])


if __name__ == "__main__":
    main()
