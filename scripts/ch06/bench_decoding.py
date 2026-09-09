from functools import partial

import torch
from common import (
    arguments,
    experiment_config,
    pixels_for,
    quality_record,
    quality_summary,
    run,
    setup,
)

from donut_camera.decoding import STRATEGIES
from donut_camera.records import write_record
from donut_camera.timing import cuda_memory, summarize_ms, timed


@torch.inference_mode()
def main():
    cli = arguments("Paired decoder-only and pixel-to-sequence benchmark")
    cli.add_argument("--warmups", type=int, default=3)
    cli.add_argument("--repetitions", type=int, default=10)
    cli.add_argument(
        "--profile-rebuilds",
        action="store_true",
        help="Separate instrumented speculative pass; excluded from main timings",
    )
    args = cli.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        cli.error("Require nonnegative warmups and positive repetitions")
    bundle, grammar, samples = setup(args)
    if bundle.device.type != "cuda":
        raise RuntimeError("Reportable decoding timing requires CUDA")
    rows = []
    quality = {strategy: [] for strategy in STRATEGIES}
    passed = True
    for index, sample in enumerate(samples):
        pixels = pixels_for(bundle, sample)
        states = bundle.model.encoder(pixels, return_dict=True)
        reference = run(bundle, grammar, args, "hf", states=states)
        call = None
        for phase in ("decoder", "model"):
            if phase == "model":
                # Drop the partial as well: it retains states from the last strategy.
                call = None
                del states
            # Rotate order deterministically across documents to reduce order bias.
            shift = index % len(STRATEGIES)
            order = STRATEGIES[shift:] + STRATEGIES[:shift]
            for strategy in order:
                inputs = (
                    {"states": states} if phase == "decoder" else {"pixels": pixels}
                )
                call = partial(run, bundle, grammar, args, strategy, **inputs)
                for _ in range(args.warmups):
                    call()
                values = []
                exact = True
                for _ in range(args.repetitions):
                    result, milliseconds = timed(call)
                    values.append(milliseconds)
                    exact = exact and torch.equal(reference.sequences, result.sequences)
                if strategy in ("greedy", "speculative"):
                    passed = passed and exact
                if phase == "decoder":
                    quality[strategy].append(quality_record(bundle, result, sample))
                rows.append(
                    {
                        "sample_index": index,
                        "phase": phase,
                        "strategy": strategy,
                        "exact_hf_sequence": exact,
                        "raw_ms": values,
                        **summarize_ms(values),
                        **result.counters(),
                        "memory": cuda_memory(call),
                    }
                )
                if args.profile_rebuilds and strategy == "speculative":
                    diagnostic = run(
                        bundle, grammar, args, strategy, profile_rebuilds=True, **inputs
                    )
                    rows[-1]["instrumented_rebuild_ms"] = diagnostic.rebuild_ms
                # The call owns the sole extra reference to visual states.
                del inputs
        write_record(
            args.output,
            config=experiment_config(bundle, args),
            measurements={
                "exactness_passed": passed,
                "documents_completed": index + 1,
                "records": rows,
                "quality": quality_summary(bundle, quality),
            },
        )
        if not passed:
            raise SystemExit(
                "Greedy/speculative parity failed: timings are not reportable"
            )
    summary = {}
    for phase in ("decoder", "model"):
        summary[phase] = {}
        for strategy in STRATEGIES:
            selected = [
                r for r in rows if r["phase"] == phase and r["strategy"] == strategy
            ]
            times = [r["median_ms"] for r in selected]
            summary[phase][strategy] = {
                "per_document_median_distribution": summarize_ms(times),
                "documents_per_second": 1000 * len(times) / sum(times),
                "tokens_per_second": 1000
                * sum(r["emitted_tokens"] for r in selected)
                / sum(times),
            }
    write_record(
        args.output,
        config=experiment_config(bundle, args),
        measurements={
            "exactness_passed": passed,
            "documents_completed": len(samples),
            "records": rows,
            "summary": summary,
            "quality": quality_summary(bundle, quality),
        },
    )


if __name__ == "__main__":
    main()
