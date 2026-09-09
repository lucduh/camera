"""Shared Chapter 6 experiment setup; no document contents are serialized."""

from pathlib import Path

import torch
from PIL import Image

from donut_camera.attention.experiments import config, load, parser
from donut_camera.data import load_samples
from donut_camera.decoding import STRATEGIES, Grammar, decode, generation_policy
from donut_camera.evaluation import parse_document
from donut_camera.metrics import extraction_metrics


def arguments(description):
    cli = parser(description)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--limit", type=int)
    cli.add_argument("--max-new-tokens", type=int, default=80)
    cli.add_argument("--max-draft", type=int, default=8)
    cli.add_argument("--no-missing-chain", action="store_true")
    return cli


def setup(args):
    if args.batch_size != 1:
        raise ValueError("Chapter 6 research paths support batch one only")
    if args.max_new_tokens < 1 or args.max_draft < 0:
        raise ValueError("Invalid generation or proposal budget")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    bundle = load(args)
    samples = load_samples(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("The experiment requires at least one document")
    return bundle, Grammar.from_bundle(bundle), samples


def pixels_for(bundle, sample):
    with Image.open(sample.image) as image:
        pixels = bundle.processor(
            image.convert("RGB"), return_tensors="pt"
        ).pixel_values
    return pixels.to(device=bundle.device, dtype=bundle.dtype)


@torch.inference_mode()
def run(
    bundle, grammar, args, strategy, *, pixels=None, states=None, profile_rebuilds=False
):
    if states is None:
        states = bundle.model.encoder(pixels, return_dict=True)
    return decode(
        bundle.model,
        states,
        bundle.decoder_start_ids(1),
        strategy=strategy,
        grammar=grammar,
        max_new_tokens=args.max_new_tokens,
        max_draft=args.max_draft,
        missing_chain=not args.no_missing_chain,
        profile_rebuilds=profile_rebuilds,
    )


def quality_record(bundle, result, sample):
    return parse_document(bundle, result.sequences[0, 1:], sample)


def quality_summary(bundle, records):
    return {
        strategy: {
            "strict": extraction_metrics(rows, bundle.task),
            "normalized": extraction_metrics(rows, bundle.task, normalized=True),
        }
        for strategy, rows in records.items()
    }


def experiment_config(bundle, args):
    policy = generation_policy(bundle.model, args.max_new_tokens)
    return {
        **config(args),
        "task_name": bundle.task.name,
        "generation_policy": "explicit greedy; no inherited processors except forced BOS/EOS",
        "forced_bos": policy.forced_bos_token_id,
        "forced_eos": policy.forced_eos_token_id,
        "strategies": STRATEGIES,
        "cache_rejection_policy": "rebuild accepted prefix excluding last token",
    }
