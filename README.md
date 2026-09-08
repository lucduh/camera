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

## Development

```bash
uv run pytest
uv run ruff format .
uv run ruff check .
```

See [PLAN.md](PLAN.md) for the experiment sequence and completion gates.
