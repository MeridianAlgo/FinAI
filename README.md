# Meridian.AI — Continual-Learning Finance LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Base Model](https://img.shields.io/badge/Base-Qwen2.5--0.5B-success.svg)](https://huggingface.co/Qwen/Qwen2.5-0.5B)
[![Training](https://img.shields.io/badge/Training-Hourly_CI-orange.svg)](https://github.com/MeridianAlgo/FinAI/actions)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-meridianal%2FFinAI-yellow.svg)](https://huggingface.co/meridianal/FinAI)
[![Version](https://img.shields.io/badge/version-1.0.2_Production-brightgreen.svg)]()

Meridian.AI is a finance-specialized language model that trains itself continuously, every hour, entirely on free GitHub Actions infrastructure. It continuously fine-tunes a **Qwen2.5-0.5B** backbone on 25+ finance and math datasets using **Elastic Weight Consolidation (EWC)** to prevent catastrophic forgetting across training sessions.

> **Status: `v1.0.0` — Production.** This is the first production-grade release. All earlier tagged builds (`v1.0.0-smollm2`, `v2.0.0-qwen`, `v5.1.0`, `v5.1.1`, `v6.0.0`) were pre-production test/research iterations and have been retired — see the [CHANGELOG](docs/CHANGELOG.md) for the full history.

**Model checkpoints:** [huggingface.co/meridianal/FinAI](https://huggingface.co/meridianal/FinAI)

> **What this is, clearly:** Qwen2.5-0.5B (a 494M-parameter causal LM from Alibaba) continuously fine-tuned on finance data via hourly GitHub Actions CI. The `meridian/` module in this repo contains a custom Sparse MoE research architecture used for experiments and smoke tests — it is **not** what is deployed in the HuggingFace checkpoint.

---

## Table of Contents

- [Why Meridian.AI](#why-meridianai)
- [Key Design Decisions](#key-design-decisions)
- [Model Specifications](#model-specifications)
- [Quick Start](#quick-start)
- [Inference](#inference)
- [Local Training](#local-training)
- [Environment Variables Reference](#environment-variables-reference)
- [CI/CD Training Pipeline](#cicd-training-pipeline)
- [Training Status & Observability](#training-status--observability)
- [Dataset Curriculum](#dataset-curriculum)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [Disclaimer](#disclaimer)

---

## Why Meridian.AI

Standard LLMs have a static knowledge cutoff. For finance — where earnings reports, Fed decisions, and market conditions change daily — this is a critical limitation.

Meridian.AI solves this with an **automated continuous training pipeline**: every hour, GitHub Actions pulls the latest checkpoint from HuggingFace, trains on fresh financial data, and pushes the updated checkpoint back. No GPUs. No cloud bills. No manual intervention.

Key constraints this design respects:
- **16 GB RAM** ceiling of free GitHub Actions ubuntu-latest runners
- **2000 CI minutes/month** free tier (public repos: unlimited)
- **No persistent storage** between runs — all state is round-tripped through HuggingFace Hub

---

## Key Design Decisions

### 1. Qwen2.5-0.5B as Training Backbone
Rather than training a model from scratch, Meridian.AI continuously fine-tunes `Qwen/Qwen2.5-0.5B` — a production-quality base model with strong pre-training on code, math, and multilingual text. This gives the model strong priors out of the box, allowing hourly fine-tuning to specialize it without training from zero.

### 2. Elastic Weight Consolidation (EWC)
After each hourly training run, the model computes the diagonal Fisher Information Matrix — a measure of which parameters were most important for tasks learned so far. The next run adds a regularization penalty for changing those parameters. This prevents the model from "forgetting" financial knowledge from earlier training sessions while absorbing new data.

### 3. Memory-Safe CPU Training
Every component is designed around the 16 GB RAM constraint:
- **AdaFactor** optimizer: eliminates the 2×-parameter-size optimizer state of Adam
- **Gradient checkpointing**: trades compute for activation memory
- **Soft RAM throttle**: dynamically truncates sequence length when memory pressure rises
- **Hard RAM guard**: emergency checkpoint + clean exit before OOM
- **Fisher threshold pruning**: only stores EWC state for parameters with Fisher value above threshold

### 4. Custom Research Architecture (meridian/)
The `meridian/` module contains a from-scratch Sparse Mixture-of-Experts Transformer (`MeridianForCausalLM`) with:
- SMoE (8 experts, top-2 per token) on alternating layers
- Grouped Query Attention (12 Q heads, 4 KV heads)
- RoPE position embeddings (theta=500,000)
- SwiGLU feed-forward blocks
- Financial Numeracy Encoding

This module is used for smoke tests and architecture experiments. It is **not** the model in the HuggingFace checkpoint — the deployed model is Qwen2.5-0.5B fine-tuned via `AutoModelForCausalLM`.

---

## Model Specifications

### Deployed Checkpoint (HuggingFace)

| Specification | Value |
|:---|:---|
| Base Model | Qwen2.5-0.5B |
| Architecture | Qwen2ForCausalLM |
| Parameters | ~494M |
| Layers | 24 |
| Hidden Size | 896 |
| Attention Heads | 14 Q / 2 KV (GQA) |
| Vocabulary | 151,643 tokens |
| Context Window | 32,768 tokens (Qwen2.5 default) |
| Training dtype | bfloat16 |
| Continual Learning | Elastic Weight Consolidation (EWC) |

### Research Architecture (meridian/)

| Specification | Value |
|:---|:---|
| Architecture | Sparse MoE Transformer |
| Layers | 14 (alternating Dense ↔ MoE) |
| Attention | GQA: 12 Q heads, 4 KV heads |
| Position Encoding | RoPE (theta=500,000) |
| Feed-Forward | SwiGLU |
| Normalization | RMSNorm |
| MoE Experts | 8 per layer, top-2 active per token |
| Vocabulary | 151,665 tokens (Qwen2.5 tokenizer) |
| Context Window | 2,048 tokens (configurable) |
| Continual Learning | Elastic Weight Consolidation |

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~2 GB disk (for model weights)

### Installation

```bash
git clone https://github.com/MeridianAlgo/FinAI.git
cd FinAI
pip install -r requirements.txt
```

### Verify Installation (Smoke Test)

Runs a tiny in-memory model to confirm the custom architecture works without downloading anything:

```bash
SMOKE_TEST=1 FAST_MODE=1 python train.py
```

Expected output: `[OK] Smoke test passed!`

---

## Inference

### From HuggingFace Hub

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "meridianal/FinAI"
subfolder = "checkpoint"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder=subfolder,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()

prompt = """### Instruction:
Explain the difference between a bond's yield to maturity and its coupon rate.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

> **Note:** `trust_remote_code=True` is **not required** for the deployed checkpoint — it is a standard Qwen2 model and loads with `AutoModelForCausalLM` directly.

### From Local Checkpoint

Download the checkpoint and run locally:

```bash
python scripts/download_and_save_hf.py
```

Then:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint")
```

See [docs/setup_and_usage.md](docs/setup_and_usage.md) for a complete inference walkthrough and recommended generation parameters.

---

## Local Training

### Full Training Run

```bash
export HF_TOKEN=your_huggingface_token
python train.py
```

The script will:
1. Pull the latest checkpoint from HuggingFace (if `HF_TOKEN` is set)
2. Load the Qwen2.5-0.5B base model (or resume from checkpoint)
3. Stream finance datasets and train for `MAX_STEPS` steps
4. Save checkpoint locally (and upload if token is present)

### Fast Debugging Mode

Runs with minimal settings (no dataset streaming, 5 steps, tiny sequences):

```bash
FAST_MODE=1 python train.py
```

### Custom Step Count

```bash
MAX_STEPS=300 BATCH_SIZE=1 python train.py
```

### Evaluate Model Quality

Compare your checkpoint against base Qwen2.5-0.5B:

```bash
python scripts/evaluate_model.py
```

---

## Environment Variables Reference

All variables are optional. CI defaults are shown in `.github/workflows/train.yml`.

### Core Training

| Variable | CI Default (v6) | Description |
|:---|:---|:---|
| `MAX_STEPS` | `150` | Gradient update steps per run |
| `TOTAL_STEPS` | `100000` | Cumulative steps across all runs (for LR schedule) |
| `BATCH_SIZE` | `1` | Samples per micro-step |
| `GRAD_ACCUM` | `4` | Micro-steps before each optimizer update |
| `LEARNING_RATE` | `5e-5` | Peak learning rate |
| `BLOCK_SIZE` | `384` | Token sequence length (lowered from 512 in v1.0.0 to keep backward-pass peak RAM under the 16 GB runner ceiling) |
| `DTYPE` | `bfloat16` | Model dtype (`bfloat16` or `float32`) |
| `OPTIMIZER` | `adafactor` | Optimizer (`adafactor` or `adamw`) |

### Memory Management

| Variable | CI Default | Description |
|:---|:---|:---|
| `HARD_RAM_GUARD` | `1` | Enable emergency save + stop at RAM ceiling |
| `MAX_RAM_GB` | `14.0` | Hard RAM limit in GB |
| `SOFT_RAM_GB` | `11.0` | Soft limit — begins sequence truncation (lowered in v1.0.0 to throttle earlier) |
| `SOFT_RAM_PCT` | `72` | Soft limit as % of total RAM |
| `MIN_THROTTLE_SEQ_LEN` | `64` | Minimum sequence length during throttle |
| `GRADIENT_CHECKPOINTING` | `1` | Trade compute for activation memory |
| `SKIP_OPTIMIZER_SAVE` | `1` | Omit optimizer state from checkpoint |

### Dataset

| Variable | CI Default | Description |
|:---|:---|:---|
| `MAX_BYTES` | `26214400` (25 MB) | Max training data per run |
| `USE_LIGHT_DATASETS` | `0` | Restrict to small/fast datasets only |

### EWC (Continual Learning)

| Variable | CI Default (v6) | Description |
|:---|:---|:---|
| `USE_EWC` | `1` | Enable Elastic Weight Consolidation |
| `EWC_LAMBDA` | `75.0` | EWC regularization strength (reduced from 500) |
| `EWC_SAMPLES` | `20` | Batches used to estimate Fisher matrix (increased from 5) |
| `SKIP_FISHER` | `0` | Skip Fisher computation (disables EWC next run) |
| `FREE_OPTIMIZER_BEFORE_FISHER` | `1` | Free optimizer RAM before Fisher computation |
| `FISHER_SEQ_LEN` | `64` | Sequence length used during Fisher estimation |
| `FISHER_THRESHOLD` | `5e-4` | Drop Fisher entries below this value (raised from 1e-4 to reduce EWC file size) |

### Paths & Misc

| Variable | Default | Description |
|:---|:---|:---|
| `CHECKPOINT_PATH` | `./checkpoint` | Local checkpoint directory |
| `TOKENIZER_ID` | `Qwen/Qwen2.5-0.5B` | HuggingFace tokenizer ID |
| `FAST_MODE` | `0` | Minimal config for quick local debugging |
| `SMOKE_TEST` | `0` | Run tiny in-memory architecture test |
| `GC_EVERY_STEPS` | `5` | Python GC frequency (steps) |
| `DEBUG_STEPS` | `0` | Print verbose per-step debug info |
| `COMET_API_KEY` | *(unset)* | Comet ML experiment tracking key |

---

## CI/CD Training Pipeline

```
Every hour (GitHub Actions cron: '0 * * * *')
│
├── Pull checkpoint from HuggingFace Hub
│     meridianal/FinAI  →  ./checkpoint/
│
├── Train (timeout: 90 minutes)
│     • Load Qwen2.5-0.5B (or resume checkpoint)
│     • Stream finance datasets (25+ sources, weighted curriculum mix)
│     • 150 AdaFactor steps with gradient checkpointing (BLOCK_SIZE=384)
│     • EWC regularization (lambda=75, 20 Fisher samples)
│     • Auto-throttle sequence length if RAM > 11.0 GB
│     • Emergency save + exit if RAM > 14.0 GB
│
├── Upload checkpoint to HuggingFace Hub
│     ./checkpoint/  →  meridianal/FinAI/checkpoint/
│
└── Sync dataset state to git
      dataset_state.json  →  main branch
```

### Failure Handling
If training encounters >50 `[ERROR]` lines or any fatal pattern (OOM, NaN explosion), the CI workflow automatically opens a GitHub Issue with the error details and a diagnostic checklist.

### Triggering a Manual Run
From the GitHub Actions tab, click **Meridian.AI Train** → **Run workflow**. You can override `MAX_STEPS` at dispatch time.

### Force Reset (Nuke & Seed)
To wipe the HuggingFace checkpoint and restart training from a fresh Qwen2.5-0.5B:

Run workflow with **force_seed: true**. This runs `scripts/seed_hf_repo.py` before training.

---

## Training Status & Observability

Because training runs unattended every hour, the project exposes several windows into what the model is doing.

### Live progress signals

| Where | What you see |
|:---|:---|
| **GitHub Actions** → *Meridian.AI Train* | Per-run logs: datasets loaded, initial/final loss, memory usage, steps completed |
| **HuggingFace** [`meridianal/FinAI`](https://huggingface.co/meridianal/FinAI) | Latest checkpoint + commit history of every hourly upload |
| **Comet ML** (`meridian-ai` workspace) | Loss curves, EWC penalty, learning rate, throughput across runs |
| **`dataset_state.json`** (git) | `processed_items` — cumulative training examples seen across all runs |

> **Continuous Comet graph (v1.0.2+):** the trainer resumes **one persistent Comet
> experiment** across hourly runs (its key lives in `checkpoint/comet_experiment.json`), so
> `loss`, `perplexity`, `ewc_loss`, `lr`, and `tokens_per_sec` form a **single continuous
> curve over `global_step`** instead of one fragment per run. Metrics are logged **every
> optimizer step** (plus an initial datapoint at the cascade check), so the graph is never
> empty even if a run is short. Set `COMET_CONTINUOUS=0` to revert to per-run experiments.

### How to read a run

Every run prints a header and a **cascade check** so you can tell at a glance whether the model is still learning:

```
==================== STARTING TRAINING RUN #1 ====================
  MERIDIAN.AI TRAINING ENGINE
  Steps: 150 | BS: 1 | Accum: 4
  LR: 5e-05 | Global step: 10127
  Memory: 2.4GB / 16.8GB (14.1% used)
  [CASCADE CHECK] Initial Loss of this run: 2.6146   ← compare across runs; a steady decline = healthy
```

- **Global step** is the cumulative optimizer-step counter; it persists across runs via `trainer_state.pt`.
- **Initial loss** should trend down over many runs (with hourly noise). Sudden spikes usually mean a new dataset region or a too-high LR.
- **Memory** is logged at the start and on every guard check — if you see `[THROTTLE]` or `[GUARD]` lines, the run hit a RAM limit and adapted.

### Current trajectory

As of the `v1.0.0` cutover the model has processed **~74,000** training examples (`processed_items: 73980`) across hourly runs, resuming from **global step ~10,127**. Reference perplexity on finance text from the most recent diagnostic was **~6.78**. Each hourly run advances the dataset cursor and pushes a fresh checkpoint, so the numbers above move continuously — check the live sources in the table for current values.

### Generation smoke test

After every upload the CI runs two finance prompts and logs token count + uniqueness ratio (see the *Generation Smoke Test* step in `train.yml`). This catches silent generation collapse before the next run builds on a broken checkpoint.

---

## Dataset Curriculum

Training data is a weighted mix of finance-focused HuggingFace datasets, streamed in real-time (no full downloads):

| Dataset | Weight | Focus |
|:---|:---|:---|
| `gbharti/finance-alpaca` | 26% | Financial Q&A instructions |
| `sujet-ai/Sujet-Finance-Instruct-177k` | 18% | High-quality finance instruction pairs |
| `nvidia/OpenMathInstruct-2` | 15% | Math reasoning (quantitative finance) |
| `HuggingFaceFW/fineweb-edu` | 12% | General knowledge foundation |
| `yahma/alpaca-cleaned` | 5% | General instruction format |
| `FinGPT/fingpt-sentiment-train` | 4% | Financial news sentiment |
| `FinanceMTEB/financial_phrasebank` | 1% | Sentiment classification |
| `FinanceMTEB/FinQA` | 1% | Financial QA pairs |
| `FinanceMTEB/TATQA` | 1% | Table-and-text QA |
| `FinanceMTEB/FOMC` | 0.8% | FOMC meeting transcripts |
| Various FinanceMTEB | ~16% | Sentiment, ESG, fraud, FLS, events, and more |

All text is formatted into the `### Instruction: / ### Response:` template before tokenization.

See [docs/training_pipeline.md](docs/training_pipeline.md) for full dataset details.

---

## Repository Structure

```
FinAI/
├── meridian/                          # Python package
│   ├── model/
│   │   ├── configuration.py           # MeridianConfig — research arch config
│   │   └── modeling.py                # MeridianForCausalLM — custom SMoE (research only)
│   ├── data/
│   │   └── pipeline.py                # Streaming dataset curriculum + DataLoader
│   └── training/
│       ├── trainer.py                 # MeridianTrainer (AdaFactor, EWC, RAM guards)
│       └── ewc.py                     # Elastic Weight Consolidation
│
├── scripts/                          # Operational + diagnostic tooling
│   ├── seed_hf_repo.py                # Nuke & reseed HuggingFace repo (used by CI seed job)
│   ├── migrate_legacy_and_seed.py     # Copy checkpoint → legacy/ and seed fresh model
│   ├── cleanup_hf_checkpoint.py       # Remove stale pytorch_model.bin from HF
│   ├── evaluate_model.py              # Evaluation: perplexity + generation quality
│   ├── diagnose_and_test.py           # Full diagnostic report (download + test)
│   ├── download_and_save_hf.py        # Download checkpoint to local directory
│   ├── hf_download_and_test.py        # Download + quick generation test
│   ├── test_generation.py            # Standalone generation sanity check
│   ├── nuke_repo.py                  # Wipe the HuggingFace repo
│   └── count_params.py                # Parameter counting utility
│
├── docs/
│   ├── architecture.md                # Detailed architecture spec (custom SMoE)
│   ├── training_pipeline.md           # Pipeline, env vars, memory management
│   ├── setup_and_usage.md             # Setup guide and inference examples
│   └── CHANGELOG.md                   # Version history and training audit
│
├── tests/
│   ├── test_model.py                  # Architecture unit tests
│   └── test_training.py               # Trainer and EWC tests
│
├── .github/
│   └── workflows/
│       ├── train.yml                  # Hourly training CI
│       ├── lint.yml                   # Ruff + Black linting
│       └── dependency-cache.yml
│
├── train.py                           # Main training entry point
├── README.md                          # This file (GitHub landing page)
├── MODEL_CARD.md                      # HuggingFace model card (uploaded by CI)
├── requirements.txt                   # Python dependencies
└── pyproject.toml                     # Ruff + Black + mypy config
```

---

## Troubleshooting

### OOM / RuntimeError during training
Reduce memory usage:
```bash
BATCH_SIZE=1 GRAD_ACCUM=4 BLOCK_SIZE=256 SOFT_RAM_GB=10.0 python train.py
```

### CI run dies with `Process completed with exit code 143`
Exit code 143 is `128 + 15` (SIGTERM) — the process tree was killed by the runner, almost always from **memory pressure during the backward pass**. The tell-tale sign is a log that stops right after `[CASCADE CHECK] Initial Loss ...` (the forward pass succeeded; the backward pass blew the RAM ceiling).

The hard/soft RAM guards only check *between* micro-steps, so they cannot catch a spike that happens *inside* a single `backward()` call. The fix is to lower the per-step activation peak:
- `BLOCK_SIZE` was reduced `512 → 384` in v1.0.0 (the v6.0.0 jump to 512 is what started triggering this on 16 GB runners).
- `SOFT_RAM_GB` was lowered `12.5 → 11.0` so sequence truncation kicks in earlier.

If you still see it, drop `BLOCK_SIZE` further (e.g. `256`) or lower `SOFT_RAM_GB` so throttling starts sooner.

### HuggingFace `429 Too Many Requests` during model download
HF rate-limits shared GitHub Actions IPs. If a run logs repeated
`HTTP Error 429 ... resolve/main/config.json` and then `OSError: couldn't connect`, the base
model couldn't be fetched. Mitigations (in place as of v1.0.1):
- The CI caches `HF_HOME` (`actions/cache`), so the Qwen base model + tokenizer download once
  and are reused — and when HF returns 429, transformers falls back to that cache instead of failing.
- `train.py` retries model/tokenizer loads with backoff and finally loads `local_files_only=True`.
- The tokenizer loads from `./checkpoint` when present, avoiding a Hub call entirely.

For local runs, set `HF_TOKEN` (authenticated requests get higher limits) and, once the base
model is cached, you can force offline mode with `TRANSFORMERS_OFFLINE=1`.

### Checkpoint architecture mismatch warning
If you see `[WARN] Checkpoint architecture mismatch (old model)`, the saved `config.json` has a `model_type` that doesn't match `qwen2`. The checkpoint will be discarded and training restarts from the base model. This is expected when switching base architectures.

### EWC shape mismatch warning
`[WARN] EWC: Dropped N params due to shape/name mismatch` appears when the model architecture changed between runs. EWC state for mismatched layers is safely dropped. Training continues normally.

### NaN loss
Usually caused by extreme learning rates or corrupted data batches. The trainer automatically skips batches with NaN loss or NaN gradients. If persistent:
```bash
LEARNING_RATE=1e-5 USE_EWC=0 python train.py
```

### EWC state file is very large (>500 MB)
Raise the Fisher threshold to prune more aggressively:
```bash
FISHER_THRESHOLD=1e-3 python train.py
```

### Slow training on CPU
Expected — these are CPU-only runners. With `BATCH_SIZE=1 BLOCK_SIZE=384 MAX_STEPS=150`, expect ~50–80 minutes per run. This fits the 90-minute CI timeout.

---

## Contributing

1. Run the test suite before submitting: `pytest tests/ -v`
2. Format with black: `black .`
3. Lint with ruff: `ruff check . --fix`
4. Keep all code pure Python — no unnecessary system dependencies
5. Submit PRs against `main` with clear commit messages (conventional commits format)

---

## Changelog

See [docs/CHANGELOG.md](docs/CHANGELOG.md) for the full version history, training audit, and issue tracker.

---

## Disclaimer

Meridian.AI is an experimental research project on continual learning for financial NLP. All model outputs are strictly for academic and research purposes. **Nothing generated by this model constitutes financial advice. Do not use outputs to make real financial decisions or execute trades.**
