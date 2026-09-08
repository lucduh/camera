import argparse
from pathlib import Path

from donut_camera.model import DEFAULT_MODEL, resolve_device
from donut_camera.tasks import load_task
from donut_camera.training import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Donut for one task")
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument(
        "--max-length",
        type=int,
        help="Defaults to max_target_length from the task configuration",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        help="Smoke-test only: stop each epoch after this many batches",
    )
    parser.add_argument(
        "--limit-validation-batches",
        type=int,
        help="Smoke-test only: validate on only this many batches",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()

    task = load_task(args.task)
    config = TrainingConfig(
        model=args.model,
        train_json=args.train,
        validation_json=args.validation,
        output=args.output,
        run_name=args.run_name,
        height=args.height,
        width=args.width,
        max_length=args.max_length or task.max_target_length,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        limit_validation_batches=args.limit_validation_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        precision=args.precision,
        seed=args.seed,
        device=resolve_device(args.device),
    )
    print(train(task, config))


if __name__ == "__main__":
    main()
