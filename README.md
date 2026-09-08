# Donut Camera

Clean experimental code for Chapters 4–6 of the internship report:

- computational characterization of Donut;
- attention acceleration;
- structured autoregressive decoding.

The project deliberately uses small scripts and plain JSON result records. It does not store confidential datasets, extracted values, or checkpoints in Git.

## Setup

```bash
uv sync
```

## Current commands

Audit a dataset before selecting sequence lengths:

```bash
uv run python scripts/audit_data.py \
  --task configs/tasks/br.toml \
  --data /secure/path/br/train.json \
  --output results/audits/br-train.json
```

Create a persistent 90/10 training and validation split when no validation set exists:

```bash
uv run python scripts/split_data.py \
  --data /secure/path/br/train.json \
  --train-output /secure/path/br/train_fit.json \
  --validation-output /secure/path/br/validation.json \
  --manifest /secure/path/br/split_manifest.json \
  --validation-fraction 0.10 --seed 42
```

Fine-tune with explicit train and validation splits:

```bash
uv run python scripts/train.py \
  --task configs/tasks/br.toml \
  --train /secure/path/br/train.json \
  --validation /secure/path/br/validation.json \
  --run-name br-1920x1440 \
  --height 1920 --width 1440
```

Evaluate a checkpoint:

```bash
uv run python scripts/evaluate.py \
  --task configs/tasks/br.toml \
  --checkpoint checkpoints/br-1920x1440/best \
  --data /secure/path/br/test.json \
  --output results/evaluation/br-1920x1440.json \
  --height 1920 --width 1440
```

Use `--debug-output debug/...` only when predictions must be inspected. Debug files can contain confidential values and are ignored by Git.

## Chapter 4 controlled experiments

The complete command sequence is in [`scripts/ch04/README.md`](scripts/ch04/README.md).

Record the architecture:

```bash
uv run python scripts/ch04/inventory.py \
  --task configs/tasks/br.toml \
  --checkpoint checkpoints/br-smoke/best \
  --output results/ch04/inventory.json
```

Run resolution-by-length inference at batch size one:

```bash
uv run python scripts/ch04/bench_inference.py \
  --task configs/tasks/br.toml \
  --checkpoint checkpoints/br-smoke/best \
  --output results/ch04/inference-resolution-length.json
```

Run the corresponding controlled training-step experiment:

```bash
uv run python scripts/ch04/bench_training.py \
  --task configs/tasks/br.toml \
  --checkpoint checkpoints/br-smoke/best \
  --output results/ch04/training-resolution-length.json
```

Batch scaling is a separate invocation, for example with
`--resolutions 1920x1440 --batch-sizes 1,2,4,8 --output-lengths 64` for
inference or `--target-lengths 64` for training.

## Development

```bash
uv run pytest
uv run ruff format .
uv run ruff check .
```

See [PLAN.md](PLAN.md) for the experiment sequence and completion gates.
