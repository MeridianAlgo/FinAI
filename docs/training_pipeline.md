# Training Pipeline

## Overview

Meridian.AI trains itself hourly using GitHub Actions on free ubuntu-latest runners (16 GB RAM, 2-core CPU). Each run:

1. Pulls the latest checkpoint from HuggingFace Hub
2. Streams fresh financial training data
3. Runs gradient updates with EWC regularization
4. Pushes the updated checkpoint back to HuggingFace Hub
5. Commits dataset state to the repository

No persistent GPU infrastructure is required. The entire pipeline runs on GitHub's free tier.

---

## Step-by-Step Execution Flow

### Step 1: Pull Checkpoint

Downloads `meridianal/FinAI/checkpoint/` from HuggingFace Hub into `./checkpoint/` on the runner.

If no checkpoint exists (first ever run, or after a nuke-and-seed), training starts fresh from `Qwen/Qwen2.5-0.5B`.

### Step 2: Lint & Format

```bash
black . --quiet
ruff check . --fix --quiet
```

Any formatting changes are committed back to the repo at the end of the run.

### Step 3: Train (`train.py`)

Full flow inside `train.py`:

1. Load model — resume checkpoint if architecture matches (`model_type` must be `qwen2` or `llama`)
2. Load tokenizer from `TOKENIZER_ID` (default: `Qwen/Qwen2.5-0.5B`)
3. Restore EWC state from `checkpoint/ewc_state.pt` if available
4. Load dataset position counter from `checkpoint/dataset_state.json` (used to derive shuffle seed)
5. Create streaming dataloader with weighted curriculum mix (shuffled with run-specific seed)
6. Train for `MAX_STEPS` gradient update steps
7. Save checkpoint (weights only — optimizer is omitted to save space)
8. Compute Fisher Information Matrix for next run's EWC
9. Save updated dataset state

### Step 4: Upload Checkpoint

Uploads `./checkpoint/` → `meridianal/FinAI/checkpoint/` on HuggingFace Hub.

Also uploads `./README.md` as the HuggingFace model card.

### Step 5: Sync State

Commits any changed files (dataset state JSON, formatting diffs) back to the `main` branch.

---

## Memory Management

GitHub's free ubuntu-latest runners provide approximately 16 GB RAM. The training loop uses several layered defenses:

### AdaFactor Optimizer

Standard AdamW stores first and second moment estimates for every parameter — roughly 2× the model size in RAM (~1 GB for 0.5B params). AdaFactor uses a factored second-moment estimate, reducing optimizer RAM by ~75%.

### Gradient Checkpointing

Instead of storing all intermediate activations for the backward pass, gradient checkpointing recomputes them from saved layer inputs. This trades ~30% extra compute for ~60% less activation memory.

### Soft RAM Throttle (`SOFT_RAM_GB` / `SOFT_RAM_PCT`)

When RAM exceeds the soft threshold, the sequence length of the current batch is truncated by 25% (down to `MIN_THROTTLE_SEQ_LEN` minimum). This immediately reduces the quadratic attention memory cost.

### Hard RAM Guard (`HARD_RAM_GUARD` / `MAX_RAM_GB`)

When RAM exceeds the hard ceiling, the trainer immediately saves a weights-only checkpoint and exits cleanly. The next run resumes from this checkpoint.

### Fisher Top-K Pruning (v6.0.1)

After training, EWC uses a **top-K-ratio pruning** strategy (`FISHER_TOP_K_RATIO=0.08`): the 8% of parameter tensors with the highest total Fisher magnitude are kept; the rest are discarded. This is much more effective than the previous `FISHER_THRESHOLD` per-element approach, which kept entire parameter tensors if ANY element exceeded the threshold — resulting in all 290 tensors being retained (~1.88 GB).

With top-K=8%:
- ~23 of 290 parameter tensors retained (the most Fisher-important weights)
- Estimated EWC state size: ~158 MB (down from 1.88 GB)
- Parameters kept represent the most "critical" weights for the EWC penalty

### Optimizer State Skipped on Save

The AdaFactor optimizer state is large (>500 MB for 0.5B params). It is not saved to disk (`SKIP_OPTIMIZER_SAVE=1`). Each run starts with a fresh optimizer, which is fine for the cosine-annealing-per-run LR schedule.

---

## Learning Rate Schedule

Each training run uses a cosine annealing schedule with linear warmup — treating each hourly run as an independent "warm restart":

```
warmup_steps = max_steps × warmup_ratio (default: 6%)

if step < warmup_steps:
    lr = peak_lr × (step + 1) / (warmup_steps + 1)
else:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    lr = min_lr + (peak_lr - min_lr) × 0.5 × (1 + cos(π × progress))
```

Default `min_lr = peak_lr × 0.1`. This means LR decays to 10% of peak by the end of each run.

---

## Dataset Streaming & Curriculum

Training data is never fully downloaded. `datasets` library streams examples record-by-record. The pipeline uses a weighted round-robin across multiple datasets:

```python
FinanceDataPipeline.DATASETS = [
    {"name": "gbharti/finance-alpaca",                  "weight": 0.26},  # Finance Q&A
    {"name": "sujet-ai/Sujet-Finance-Instruct-177k",    "weight": 0.18},  # Finance instruct
    {"name": "nvidia/OpenMathInstruct-2",               "weight": 0.15},  # Math reasoning
    {"name": "HuggingFaceFW/fineweb-edu",               "weight": 0.12},  # General knowledge
    {"name": "yahma/alpaca-cleaned",                    "weight": 0.05},  # Instruction format
    # + FinGPT, nickmuchi/financial-classification, 20 FinanceMTEB datasets
]
```

Data is formatted as instruction-response pairs:

```
### Instruction:
{instruction or prompt}

### Response:
{text or label}{eos_token}
```

The pipeline tracks `processed_items` (total examples seen across all runs) and derives a **shuffle seed** (`processed_items % 100_000`) for each dataset. This ensures each hourly run samples a different region of every dataset without the sequential `dataset.skip(N)` overhead that was previously used — which required downloading thousands of items before yielding training examples (causing ~70-minute wasted time per run when `processed_items` was large).

Per-run data intake is capped at `MAX_BYTES` (default: 25 MB as of v6.0.0, previously 15 MB) to keep each run's training data consistent regardless of step count.

---

## Checkpoint Structure

After a training run, `./checkpoint/` contains:

```
checkpoint/
├── config.json           # Model config (Qwen2 architecture)
├── model.safetensors     # Model weights
├── tokenizer.json        # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── trainer_state.pt      # global_step, best_loss (no optimizer)
├── ewc_state.pt          # Fisher diag + prev params (for next run's EWC)
└── dataset_state.json    # processed_items count
```

### Resuming

On startup, `train.py`:
1. Checks `checkpoint/config.json` for `model_type` — rejects if not `qwen2` or `llama`
2. Loads model via `AutoModelForCausalLM.from_pretrained(checkpoint_path)`
3. Loads `trainer_state.pt` to restore `global_step` and `best_loss`
4. Loads `ewc_state.pt`, validates tensor shapes against current model

---

## Failure Handling

The CI workflow monitors training output for error patterns:

| Pattern | Action |
|:---|:---|
| >50 `[ERROR]` lines | Creates GitHub Issue with error summary |
| `CUDA out of memory` | Creates GitHub Issue (shouldn't happen on CPU) |
| `Loss is NaN` | Creates GitHub Issue |
| `RuntimeError` | Creates GitHub Issue |
| Exit code non-zero | Checkpoint still uploaded (training may have partially succeeded) |

Individual NaN batches are skipped automatically. Training continues until `MAX_STEPS` is reached.

### Exit code 143 (SIGTERM) — backward-pass OOM

`Process completed with exit code 143` means the runner SIGTERM-killed the whole step process tree (`128 + 15`). It is **not** a clean exit — the workflow's `exit 0` safety net never runs because bash itself is killed. The signature in the log is output that stops immediately after:

```
[CASCADE CHECK] Initial Loss of this run: 2.6146
```

i.e. the forward pass completed but the **backward pass** pushed peak RAM past the ~16 GB runner ceiling. The hard/soft RAM guards only evaluate *between* micro-steps, so they cannot intercept a spike inside a single `backward()`.

**Mitigation (applied in v1.0.0):** `BLOCK_SIZE 512 → 384` (lower activation peak) and `SOFT_RAM_GB 12.5 → 11.0` (earlier sequence truncation). If it recurs, lower `BLOCK_SIZE` to `256`.

---

## Environment Variables — Full Reference

See [README.md](../README.md#environment-variables-reference) for the complete table.

### Quick Reference (CI Defaults — v1.0.0 Production)

```bash
MAX_STEPS=150          # Steps per run
BATCH_SIZE=1           # Micro-batch size
GRAD_ACCUM=4           # Effective batch size = 4
BLOCK_SIZE=384         # Sequence length (lowered 512→384 in v1.0.0 to fit 16 GB backward peak)
LEARNING_RATE=5e-5     # Peak LR
DTYPE=bfloat16         # Model precision
OPTIMIZER=adafactor    # Memory-efficient optimizer
USE_EWC=1              # Enable continual learning
EWC_LAMBDA=75.0        # EWC regularization
EWC_SAMPLES=20         # Fisher estimation batches
HARD_RAM_GUARD=1       # Emergency save at 14.0 GB
MAX_RAM_GB=14.0
SOFT_RAM_GB=11.0       # Begin sequence throttle (lowered 12.5→11.0 in v1.0.0)
SOFT_RAM_PCT=72
MAX_BYTES=26214400     # 25 MB data per run
GRADIENT_CHECKPOINTING=1
SKIP_OPTIMIZER_SAVE=1
FREE_OPTIMIZER_BEFORE_FISHER=1
FISHER_SEQ_LEN=64
FISHER_THRESHOLD=5e-4
FISHER_TOP_K_RATIO=0.08  # Keep top 8% of params by Fisher magnitude (~158 MB EWC state)
```

---

## Running Training Locally

### With HuggingFace checkpoint sync

```bash
export HF_TOKEN=hf_xxxxx
export CHECKPOINT_PATH=./checkpoint
python train.py
```

### Without HuggingFace (offline, fresh start)

```bash
MAX_STEPS=50 BATCH_SIZE=1 GRAD_ACCUM=2 BLOCK_SIZE=128 python train.py
```

### Smoke test (no external dependencies)

```bash
SMOKE_TEST=1 FAST_MODE=1 python train.py
```
