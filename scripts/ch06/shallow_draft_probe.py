"""Measure truncated-draft agreement and speculative decoding mechanics."""

import argparse
import time
from pathlib import Path

import torch
from common import experiment_config, pixels_for, setup

from donut_camera.decoding import decode
from donut_camera.records import write_record
from donut_camera.shallow_draft import load_draft, speculative_decode, visual_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft-checkpoint")
    parser.add_argument("--draft-depth", type=int, default=1)
    parser.add_argument("--preset", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-draft", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    bundle, _grammar, samples = setup(args)
    samples = samples[: args.limit]
    draft = load_draft(bundle.model, args.draft_checkpoint, args.draft_depth)
    rows = []
    for index, sample in enumerate(samples):
        pixels = pixels_for(bundle, sample)
        with torch.inference_mode():
            states = bundle.model.encoder(pixels, return_dict=True)
            memory = visual_memory(bundle.model, states)
            prompt = bundle.decoder_start_ids(1)
            reference = decode(
                bundle.model,
                states,
                prompt,
                strategy="greedy",
                max_new_tokens=args.max_new_tokens,
            )
            started = time.perf_counter()
            candidate, stats = speculative_decode(
                bundle.model,
                draft,
                states,
                prompt,
                max_new_tokens=args.max_new_tokens,
                max_draft=args.max_draft,
                confidence=args.confidence,
            )
            if bundle.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            draft_logits = draft(
                input_ids=prompt,
                encoder_hidden_states=memory,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1]
            target_logits = bundle.model.decoder(
                input_ids=prompt,
                encoder_hidden_states=memory,
                use_cache=False,
                return_dict=True,
            ).logits[:, -1]
        rows.append(
            {
                "sample_index": index,
                "exact_target_greedy": torch.equal(
                    reference.sequences, candidate.sequences
                ),
                "draft_target_top1_agreement_at_prompt": bool(
                    draft_logits.argmax(-1).item() == target_logits.argmax(-1).item()
                ),
                "elapsed_ms": elapsed_ms,
                "draft_calls": stats.calls,
                "rounds": len(stats.rounds),
                "proposed": stats.proposed,
                "accepted": stats.accepted,
                "acceptance_rate": stats.accepted / stats.proposed
                if stats.proposed
                else None,
                "tokens_per_target_call": candidate.counters()[
                    "tokens_per_target_call"
                ],
            }
        )
    write_record(
        args.output,
        config={
            **experiment_config(bundle, args),
            "draft_depth": args.draft_depth,
            "draft_checkpoint": bool(args.draft_checkpoint),
        },
        measurements={"records": rows},
    )


if __name__ == "__main__":
    main()
