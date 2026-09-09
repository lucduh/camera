"""Shared CLI and data setup; result records never include document contents."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from donut_camera.attention import PRESETS, apply_attention
from donut_camera.data import DonutDataset, load_samples
from donut_camera.model import DEFAULT_MODEL, load_bundle, resolve_device
from donut_camera.tasks import load_task


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--task", type=Path, required=True)
    result.add_argument("--checkpoint", default=DEFAULT_MODEL)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--preset", choices=PRESETS, default="sdpa")
    result.add_argument("--height", type=int, default=1920)
    result.add_argument("--width", type=int, default=1440)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--length", type=int, default=56)
    result.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    result.add_argument("--device")
    result.add_argument("--seed", type=int, default=42)
    return result


def load(args, *, training=False, baseline=False):
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
    bundle = load_bundle(
        args.checkpoint,
        task=load_task(args.task),
        device=device,
        dtype=args.dtype,
        training=training,
    )
    bundle.set_resolution(args.height, args.width)
    apply_attention(bundle.model, "baseline" if baseline else args.preset)
    return bundle


def config(args):
    # Only public experiment settings, never private checkpoint/data paths.
    try:
        fa4_version = version("flash-attn-4")
    except PackageNotFoundError:
        fa4_version = None
    return {
        "flash_attn_4": fa4_version,
        **{
            k: v
            for k, v in vars(args).items()
            if k not in {"checkpoint", "data", "task", "output"}
        },
    }


def synthetic(bundle, args):
    generator = torch.Generator(device=bundle.device).manual_seed(args.seed)
    pixels = torch.randn(
        args.batch_size,
        3,
        args.height,
        args.width,
        device=bundle.device,
        dtype=bundle.dtype,
        generator=generator,
    )
    labels = torch.randint(
        4,
        bundle.model.decoder.config.vocab_size,
        (args.batch_size, args.length),
        device=bundle.device,
        generator=generator,
    )
    return pixels, labels


def real_batches(bundle, args):
    dataset = DonutDataset(
        load_samples(args.data), bundle.processor, bundle.task, args.length
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


def difference(reference, candidate):
    a, b = reference.float().flatten(), candidate.float().flatten()
    return {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "max_absolute_error": (a - b).abs().max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        "exact": torch.equal(reference, candidate),
    }
