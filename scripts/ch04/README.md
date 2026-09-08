# Chapter 4 runbook

All commands are run from the `donut-camera/` directory. Final report numbers
must use the same H100, software lockfile, eager attention, BF16 precision, and
fine-tuned BR checkpoint.

Set these paths for the current environment:

```bash
TASK=configs/tasks/br.toml
TRAIN=/secure/path/br/train_fit.json
VALIDATION=/secure/path/br/validation.json
TEST=/secure/path/br/test.json
# Use this while validating commands. Replace it after complete fine-tuning.
CHECKPOINT=checkpoints/br-smoke/best
mkdir -p results/ch04
```

The smoke checkpoint is sufficient to test benchmark code because it has the
same architecture. Final report records must be regenerated after setting
`CHECKPOINT=checkpoints/br-1920x1440/best`.

## 1. Architecture inventory

```bash
uv run python scripts/ch04/inventory.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output results/ch04/inventory.json
```

This records parameter counts, stage dimensions, token/window counts, and
analytical encoder-block MAC estimates. The vocabulary projection is tied to
the decoder embedding and is not an additional parameter group.

## 2. Complete BR fine-tuning

Select the physical batch size using the controlled batch benchmark below,
then freeze it before the final run. Replace the three shell values if the
canonical protocol changes.

```bash
BATCH_SIZE=1
EPOCHS=30
LEARNING_RATE=3e-4

uv run python scripts/train.py \
  --task "$TASK" \
  --train "$TRAIN" \
  --validation "$VALIDATION" \
  --output checkpoints \
  --run-name br-1920x1440 \
  --height 1920 \
  --width 1440 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --workers 4 \
  --learning-rate "$LEARNING_RATE" \
  --warmup-steps 100 \
  --precision bf16 \
  --seed 42
```

`train.json` records training, validation, and combined epoch times, total
fine-tuning time, best epoch, throughput, and peak allocated GPU memory. After
this run, select the final checkpoint for every reportable command:

```bash
CHECKPOINT=checkpoints/br-1920x1440/best
```

## 3. Real BR quality and inference profile

Run batch-one inference over the complete test set. Profile mode deliberately
uses no DataLoader workers so preprocessing, transfer, encoder, decoder, and
parsing are measured separately for every document.

```bash
uv run python scripts/evaluate.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --data "$TEST" \
  --output results/ch04/br-real-profile.json \
  --height 1920 \
  --width 1440 \
  --batch-size 1 \
  --workers 0 \
  --max-new-tokens 80 \
  --dtype bf16 \
  --profile
```

The same record contains strict and normalized F1, structured validity, output
lengths, raw per-document timings, medians, p95 values, and memory.

## 4. Controlled inference: resolution and output length

Smoke test:

```bash
uv run python scripts/ch04/bench_inference.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output results/ch04/inference-smoke.json \
  --resolutions 1280x960 \
  --batch-sizes 1 \
  --output-lengths 16 \
  --warmups 1 \
  --repetitions 2
```

Final resolution/length sweep:

```bash
uv run python scripts/ch04/bench_inference.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output results/ch04/inference-resolution-length.json \
  --resolutions 1280x960,1920x1440,2560x1920 \
  --batch-sizes 1 \
  --output-lengths 16,32,56,64,80 \
  --warmups 5 \
  --repetitions 20
```

## 5. Controlled inference: batch scaling

```bash
uv run python scripts/ch04/bench_inference.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output results/ch04/inference-batch-scaling.json \
  --resolutions 1920x1440 \
  --batch-sizes 1,2,4,8 \
  --output-lengths 56 \
  --warmups 5 \
  --repetitions 20
```

OOM configurations are valid observations and are recorded rather than hidden.

## 6. Controlled training: resolution and target length

```bash
uv run python scripts/ch04/bench_training.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output results/ch04/training-resolution-length.json \
  --resolutions 1280x960,1920x1440,2560x1920 \
  --batch-sizes 1 \
  --target-lengths 32,56,64,80 \
  --warmups 3 \
  --repetitions 10
```

## 7. Controlled training: batch scaling

```bash
uv run python scripts/ch04/bench_training.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output results/ch04/training-batch-scaling.json \
  --resolutions 1920x1440 \
  --batch-sizes 1,2,4,8 \
  --target-lengths 56 \
  --warmups 3 \
  --repetitions 10
```

The optimizer uses a zero learning rate in controlled timing so every phase is
executed while model weights remain fixed.

## 8. Encoder-stage and decoder-component profile

Run one representative operating point:

```bash
uv run python scripts/ch04/profile_components.py \
  --task "$TASK" \
  --checkpoint "$CHECKPOINT" \
  --output results/ch04/components-1920x1440.json \
  --height 1920 \
  --width 1440 \
  --batch-size 1 \
  --output-length 56 \
  --warmups 5 \
  --repetitions 20
```

The record contains uninstrumented reference timings as well as CUDA-event
measurements for patch embedding, Swin stages 0--3, decoder layers 0--3, token
embedding, and vocabulary projection. Use uninstrumented timings for absolute
latency and component events only for attribution.

## 9. Analysis

Open `analysis/chapter04.ipynb` after copying the JSON records into
`results/ch04/`. The notebook performs no model execution.

The shifted-window mask anomaly remains explicitly TBD. Its eventual command
will be added here only after the isolated reproduction is implemented and
validated.
