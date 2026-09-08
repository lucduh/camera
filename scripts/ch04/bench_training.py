import argparse
import gc
import statistics
from functools import partial
from pathlib import Path

import torch

from donut_camera.model import DEFAULT_MODEL, autocast, load_bundle, resolve_device
from donut_camera.records import write_record
from donut_camera.tasks import load_task


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_resolutions(value: str) -> list[tuple[int, int]]:
    return [tuple(map(int, item.split("x"))) for item in value.split(",")]


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for phase in (
        "forward_ms",
        "backward_ms",
        "gradient_clip_ms",
        "optimizer_ms",
        "total_ms",
    ):
        values = [row[phase] for row in rows]
        summary[phase] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def timed_step(model, optimizer, pixels, labels, *, device: str, precision: str):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    torch.cuda.synchronize()
    events[0].record()
    with autocast(device, precision):
        loss = model(pixel_values=pixels, labels=labels).loss
    events[1].record()
    loss.backward()
    events[2].record()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    events[3].record()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    events[4].record()
    torch.cuda.synchronize()
    return {
        "forward_ms": events[0].elapsed_time(events[1]),
        "backward_ms": events[1].elapsed_time(events[2]),
        "gradient_clip_ms": events[2].elapsed_time(events[3]),
        "optimizer_ms": events[3].elapsed_time(events[4]),
        "total_ms": events[0].elapsed_time(events[4]),
        "loss": loss.item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled Donut training-step scaling benchmark"
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolutions", default="1280x960,1920x1440,2560x1920")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--target-lengths", default="32,56,64,80")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if not device.startswith("cuda"):
        raise RuntimeError("The reportable training benchmark requires CUDA")
    task = load_task(args.task)
    bundle = load_bundle(
        args.checkpoint, task=task, device=device, dtype="fp32", training=True
    )
    model = bundle.model
    # Zero learning rate keeps model weights fixed while preserving optimizer work.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    records = []

    for height, width in parse_resolutions(args.resolutions):
        bundle.set_resolution(height, width)
        for batch_size in parse_ints(args.batch_sizes):
            for target_length in parse_ints(args.target_lengths):
                try:
                    pixels = torch.randn(
                        batch_size,
                        model.encoder.config.num_channels,
                        height,
                        width,
                        device=device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    labels = torch.randint(
                        0,
                        model.decoder.config.vocab_size,
                        (batch_size, target_length),
                        device=device,
                        generator=generator,
                    )
                    step = partial(
                        timed_step,
                        model,
                        optimizer,
                        pixels,
                        labels,
                        device=device,
                        precision=args.precision,
                    )
                    for _ in range(args.warmups):
                        step()
                    rows = [step() for _ in range(args.repetitions)]

                    torch.cuda.reset_peak_memory_stats()
                    start_allocated = torch.cuda.memory_allocated()
                    memory_row = step()
                    peak_allocated = torch.cuda.max_memory_allocated()
                    records.append(
                        {
                            "status": "ok",
                            "resolution": [height, width],
                            "batch_size": batch_size,
                            "target_length": target_length,
                            "raw": rows,
                            "summary": summarize(rows),
                            "memory": {
                                "start_allocated_mb": start_allocated / 1024**2,
                                "peak_allocated_mb": peak_allocated / 1024**2,
                                "incremental_peak_mb": (
                                    peak_allocated - start_allocated
                                )
                                / 1024**2,
                                "measurement_step": memory_row,
                            },
                        }
                    )
                    del pixels, labels
                except torch.OutOfMemoryError as error:
                    optimizer.zero_grad(set_to_none=True)
                    records.append(
                        {
                            "status": "oom",
                            "resolution": [height, width],
                            "batch_size": batch_size,
                            "target_length": target_length,
                            "error": str(error),
                        }
                    )
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

    path = write_record(
        args.output,
        config={
            "task": task.name,
            "checkpoint": Path(args.checkpoint).name,
            "resolutions": args.resolutions,
            "batch_sizes": args.batch_sizes,
            "target_lengths": args.target_lengths,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "precision": args.precision,
            "device": device,
            "decoder_attention": model.decoder.config._attn_implementation,
            "mask_cache": False,
            "optimizer": "AdamW(lr=0, weight_decay=0)",
            "grad_clip": 1.0,
        },
        measurements={"records": records},
    )
    print(path)


if __name__ == "__main__":
    main()
