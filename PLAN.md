# Donut Camera experiment plan

## Rules

- BR is the only implementation and execution focus for the current phase.
- Report tables retain KPID and KPD rows, left empty until later coverage runs.
- Standard resolutions are 1280x960, 1920x1440, and 2560x1920.
- Use explicit train, validation, and test splits.
- Save raw measurements and environment metadata in JSON.
- Keep predictions and identifiable paths out of reportable records.
- Record unsupported configurations and OOMs; never silently fall back.
- Stabilize each chapter before implementing the next one.

## Foundation

- [x] Task-specific schemas
- [x] Dataset loading and fixed-schema target serialization
- [x] Data/schema audit command
- [x] Metrics in which a wrong value contributes FP and FN
- [x] Minimal model loading and resolution configuration
- [x] Explicit-split training command
- [x] Evaluation command with sanitized per-document records
- [x] Audit BR training schema and untruncated target lengths
- [x] Classify BR duplicate fields and adopt a first-occurrence policy
- [x] Verify BR test schema and target lengths
- [x] Create and freeze the BR validation split (10%, seed 42)
- [ ] Complete a BR smoke train/save/reload/evaluate run
- [ ] Freeze canonical fine-tuning hyperparameters

## Current BR audit

- 3,773 training documents
- no empty annotations
- no unknown fields
- median target length: 56 tokens
- maximum target length: 67 tokens
- six targets exceed 64 tokens
- 1,071 documents contain at least one duplicate field
- most conflicting groups are rare; `numero_da_nota` has 124
- BR test targets have fewer conflicting groups and a maximum of 64 tokens
- training and generation limit: 80 tokens, determined by the training maximum

Repeated annotations use an explicit first-occurrence policy, matching source
JSON order. This is recorded in the BR task configuration and experiment
metadata. The 124 conflicting `numero_da_nota` groups remain a dataset-quality
limitation to mention in the report, not a blocker for profiling.

## Chapter 4 — Computational characterization

1. Architecture, parameter, token, window, and target-length inventory.
2. Baseline fine-tuning and quality at the three resolution tiers.
3. Real batch-one inference decomposition: preprocessing, transfer, encoder,
   decoder, parsing, and wall time.
4. Controlled resolution x output-length inference sweep.
5. Batch-size latency, throughput, and memory sweep.
6. Encoder-stage and decoder-component CUDA-event profile.
7. Controlled training-step sweep: forward, backward, optimizer, and memory.
8. Real training-pipeline profile including DataLoader wait and transfer.
9. Historical/current/cached shifted-window mask reproduction.

BR receives the complete grid. KPID and KPD remain represented by empty report
table rows during this phase; their future runs only need quality and selected
confirmation measurements because the model architecture is unchanged.

## Chapter 5 — Attention acceleration

1. Implement mask caching, encoder SDPA, decoder SDPA controls, and optional FA4
   as independent switches.
2. Audit encoder outputs, logits, sequences, losses, and gradients.
3. Benchmark Swin-window, decode, and prefill/training kernel shapes.
4. Benchmark incremental full-model configurations.
5. Compare training-step latency and memory.
6. Run high-resolution BF16 stability ablations.

## Chapter 6 — Structured decoding

1. Implement a custom greedy KV-cache loop with exact Hugging Face parity.
2. Build a minimal ordered-field schema from each task configuration.
3. Evaluate constrained/template decoding.
4. Implement and evaluate a clearly defined speculative baseline.
5. Implement exact-verification and forced schema-aware block modes.
6. Report model calls, forced/proposed/accepted tokens, latency, validity, and F1
   on BR, with KPID/KPD confirmation.

## Completion gates

A benchmark result is reportable only if:

- its model, task, data split, resolution, batch, length, dtype, software, GPU,
  seed, and Git revision are recorded;
- CUDA timings are synchronized and include raw repetitions;
- memory distinguishes starting allocation from incremental peak;
- correctness has been checked at the level required by the experiment;
- the canonical command is documented;
- the analysis reads saved records rather than copied console output.
