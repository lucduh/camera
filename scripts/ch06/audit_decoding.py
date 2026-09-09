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


@torch.inference_mode()
def main():
    args = arguments(
        "Audit decoding equality and extraction quality without saving text"
    ).parse_args()
    bundle, grammar, samples = setup(args)
    rows = []
    quality = {strategy: [] for strategy in STRATEGIES}
    passed = True
    for index, sample in enumerate(samples):
        pixels = pixels_for(bundle, sample)
        states = bundle.model.encoder(pixels, return_dict=True)
        reference = run(bundle, grammar, args, "hf", states=states)
        for strategy in STRATEGIES:
            result = (
                reference
                if strategy == "hf"
                else run(bundle, grammar, args, strategy, states=states)
            )
            exact = torch.equal(reference.sequences, result.sequences)
            quality[strategy].append(quality_record(bundle, result, sample))
            rows.append(
                {
                    "sample_index": index,
                    "strategy": strategy,
                    "exact_hf_sequence": exact,
                    **result.counters(),
                }
            )
            if strategy in ("greedy", "speculative"):
                passed = passed and exact
    write_record(
        args.output,
        config=experiment_config(bundle, args),
        measurements={
            "exactness_passed": passed,
            "documents": len(samples),
            "records": rows,
            "quality": quality_summary(bundle, quality),
        },
    )
    if not passed:
        raise SystemExit(
            "Exactness audit failed; numeric diagnostics saved, no token contents written"
        )


if __name__ == "__main__":
    main()
