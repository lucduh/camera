"""Distill a shallow mBART decoder from a frozen Donut target."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from donut_camera.attention.experiments import load
from donut_camera.data import DonutDataset, load_samples
from donut_camera.records import write_record
from donut_camera.shallow_draft import (
    distillation_loss,
    load_draft,
    shifted_inputs,
    visual_memory,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--draft-depth", type=int, default=1)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")

    # load() applies the selected attention preset to the target; distillation
    # defaults to eager to keep the draft experiment independent of Chapter 5.
    args.preset = "baseline"
    bundle = load(args, training=True)
    target = bundle.model.eval()
    draft = load_draft(target, depth=args.draft_depth).train()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    dataset = DonutDataset(
        load_samples(args.train), bundle.processor, bundle.task, args.max_length
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(draft.parameters(), lr=args.learning_rate)
    losses = []

    for step in tqdm(range(1, args.steps + 1), desc="distill draft"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        pixels = batch["pixel_values"].to(bundle.device, dtype=bundle.dtype)
        labels = batch["labels"].to(bundle.device)
        with torch.no_grad():
            states = target.encoder(pixels, return_dict=True)
            memory = visual_memory(target, states)
            decoder_inputs = shifted_inputs(
                labels,
                target.config.decoder_start_token_id,
                bundle.processor.tokenizer.pad_token_id,
            )
            teacher = target.decoder(
                input_ids=decoder_inputs,
                encoder_hidden_states=memory,
                use_cache=False,
                return_dict=True,
            ).logits
        student = draft(
            input_ids=decoder_inputs,
            encoder_hidden_states=memory,
            use_cache=False,
            return_dict=True,
        ).logits
        loss = distillation_loss(
            student,
            teacher,
            labels,
            temperature=args.temperature,
            ce_weight=args.ce_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    args.output.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(args.output)
    write_record(
        args.record,
        config={
            "draft_depth": args.draft_depth,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "ce_weight": args.ce_weight,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        measurements={
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_median_last_20": sorted(losses[-20:])[len(losses[-20:]) // 2],
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
