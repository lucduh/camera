import time

import torch

from donut_camera.attention import fallback_calls
from donut_camera.attention.experiments import (
    config,
    load,
    parser,
    real_batches,
    synthetic,
)
from donut_camera.model import autocast
from donut_camera.records import write_record
from donut_camera.timing import summarize_ms, synchronize


def first_nonfinite(named_tensors):
    for name, tensor in named_tensors:
        if tensor is not None and not torch.isfinite(tensor).all():
            return name
    return None


def main():
    cli = parser("Real-update stability audit; run each preset in a fresh process")
    cli.add_argument("--data", type=str)
    cli.add_argument("--steps", type=int, default=20)
    cli.add_argument("--learning-rate", type=float, default=3e-4)
    cli.add_argument("--weight-decay", type=float, default=0.01)
    cli.add_argument("--grad-clip", type=float, default=1.0)
    cli.add_argument("--diagnostic", action="store_true")
    args = cli.parse_args()
    if args.dtype == "fp16":
        cli.error(
            "Stability protocol supports bf16 autocast or fp32, not unscaled fp16"
        )
    bundle = load(args, training=True)
    model = bundle.model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.data:
        batches = real_batches(bundle, args)
        iterator = iter(batches)
    else:
        pixels, labels = synthetic(bundle, args)
    diagnostics = []
    handles = []
    if args.diagnostic:

        def hook(name, module, inputs, output):
            tensors = output if isinstance(output, tuple) else (output,)
            if any(
                isinstance(t, torch.Tensor) and not torch.isfinite(t).all()
                for t in tensors
            ):
                diagnostics.append(name)

        from functools import partial

        for name, module in model.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(partial(hook, name)))
    rows = []
    failure = None
    try:
        for step in range(1, args.steps + 1):
            if args.data:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(batches)
                    batch = next(iterator)
                pixels, labels = (
                    batch[key].to(bundle.device) for key in ("pixel_values", "labels")
                )
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            if bundle.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with autocast(bundle.device, args.dtype):
                loss = model(pixel_values=pixels, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            # These checks are outside step timing; diagnostic hooks are not.
            checks = {
                "loss": first_nonfinite([("loss", loss)]),
                "gradients": first_nonfinite(
                    (n, p.grad) for n, p in model.named_parameters()
                ),
                "parameters": first_nonfinite(model.named_parameters()),
                "optimizer": first_nonfinite(
                    (f"{i}.{key}", value)
                    for i, state in enumerate(optimizer.state.values())
                    for key, value in state.items()
                    if isinstance(value, torch.Tensor)
                ),
            }
            row = {
                "step": step,
                "loss": loss.item() if torch.isfinite(loss) else None,
                "step_ms": elapsed,
                "checks": checks,
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
                if bundle.device.type == "cuda"
                else None,
            }
            rows.append(row)
            if any(value is not None for value in checks.values()):
                failure = step
                break
    finally:
        for handle in handles:
            handle.remove()
    write_record(
        args.output,
        config=config(args),
        measurements={
            "steps": rows,
            "first_nonfinite_step": failure,
            "first_nonfinite_modules": diagnostics[:20],
            "timing_excludes_first_step": summarize_ms(
                [r["step_ms"] for r in rows[1:]]
            ),
            "timing_instrumented": args.diagnostic,
            "cudnn_fallback_calls": fallback_calls(model),
        },
    )


if __name__ == "__main__":
    main()
