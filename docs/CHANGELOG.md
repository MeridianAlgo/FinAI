# Meridian.AI — Changelog

All notable changes across training versions are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **Versioning note (2026-06-05):** `v1.0.0` is the first **production** release.
> Every version below it (`6.0.1`, `6.0.0`, `5.1.x`, `5.0.0`, `4.0.0`, `3.0.0`, `2.0.0`,
> and the original `0.1.0` prototype) was a **pre-production test / research iteration**.
> The corresponding git tags (`v1.0.0-smollm2`, `v2.0.0-qwen`, `v5.1.0`, `v5.1.1`,
> `v6.0.0`) have been **deleted** — they were experimental checkpoints, not releases.
> History is preserved below for the engineering audit trail.

---

## [1.0.1] — 2026-06-05 — Hotfix

> **Hotfix:** the hourly CI run kept dying with HuggingFace **HTTP 429 (Too Many Requests)**
> while downloading the `Qwen/Qwen2.5-0.5B` base model.

### Fixed — CI training killed by HF 429 rate limiting

- **Root cause:** when the checkpoint pull didn't land a local `model.safetensors`,
  `train.py` re-downloaded the **base model from HuggingFace on every run**. Shared GitHub
  Actions IPs get aggressively rate-limited, so `config.json` HEAD requests returned 429,
  exhausted the 5 built-in retries, and the run crashed with `OSError: couldn't connect`.
  Nothing persisted the base model between runs, so every hour hammered the Hub again.
- **Fixes:**
  - **Persistent HF cache** — the workflow now caches `HF_HOME` (`actions/cache`, key
    `hf-cache-qwen2.5-0.5b-v1`). The base model + tokenizer download once and are reused;
    when HF returns 429, transformers transparently falls back to the cached copy instead
    of failing.
  - **Resilient loaders in `train.py`** — base-model and tokenizer loads now pass `HF_TOKEN`,
    retry with exponential backoff (`hf_load_with_retry`), and finally fall back to the local
    cache (`local_files_only=True`).
  - **Tokenizer prefers the checkpoint** — when a tokenizer is present in `./checkpoint`, it
    loads from there (zero Hub calls) instead of re-fetching from Qwen every run.
  - **Checkpoint pull retries** — `snapshot_download` now retries 3× with backoff.
  - Added `HF_HUB_DOWNLOAD_TIMEOUT=60` to the training job.

---

## [1.0.0] — 2026-06-05 — **Production**

> **First production release.** Promotes the continually-trained Qwen2.5-0.5B finance model
> out of the test phase, and fixes the hourly CI run that was dying with `exit code 143`.

### Fixed — CI training killed with exit code 143 (SIGTERM)

- **Root cause:** the hourly run was being SIGTERM-killed (`128 + 15 = 143`) by the GitHub
  Actions runner during the **first backward pass**. The log signature was output that
  stopped immediately after `[CASCADE CHECK] Initial Loss of this run: ...` — the forward
  pass succeeded, but backprop pushed peak RAM past the ~16 GB runner ceiling. Because the
  kill hit the whole step process tree, the workflow's `exit 0` safety net never ran, so the
  step surfaced 143 instead of a clean early-stop.
- **Why the guards missed it:** `_memory_guard()` / soft-throttle only evaluate *between*
  micro-steps, so they cannot intercept a memory spike that happens *inside* a single
  `backward()` call. The `BLOCK_SIZE` jump `256 → 512` in v6.0.0 is what started tripping
  this intermittently on 16 GB runners.
- **Fix:** lowered the per-step activation peak and made throttling more aggressive:
  - `BLOCK_SIZE` `512 → 384`
  - `SOFT_RAM_GB` `12.5 → 11.0` (sequence truncation kicks in earlier)
  - `MAX_RAM_GB` `14.5 → 14.0` (more headroom before the hard guard)

### Changed

- Version banner in `train.py` updated to `v1.0.0 (Production)`.
- README version badge → `1.0.0 Production`; added a **Training Status & Observability**
  section (live signals, how to read a run, current trajectory) and an exit-143
  troubleshooting entry.

### Docs & Repository Layout

- `README.md`, `docs/training_pipeline.md`, and `docs/setup_and_usage.md` updated to v1.0.0
  defaults.
- **Split the GitHub README from the HuggingFace model card.** The root `README.md` no
  longer carries YAML frontmatter (which GitHub rendered as an ugly metadata table at the
  top of the repo page). A dedicated `MODEL_CARD.md` now carries the frontmatter and is what
  CI uploads to HuggingFace as the model card.
- **Moved `CHANGELOG.md` → `docs/CHANGELOG.md`** so all documentation lives under `docs/`.
- **Removed the `examples/` directory** — the inference/usage snippets in `README.md` and
  `docs/setup_and_usage.md` cover the same ground.
- `scripts/` retained: the CI seed job runs `scripts/seed_hf_repo.py`, and the others are
  referenced operational/diagnostic tooling.

---

## [6.0.1] — 2026-05-26

> **Hotfix batch:** Four critical fixes discovered after v6.0.0 deploy.

### Changes

#### Critical — Training Throughput (was 5 steps/run, target 120+)
- **Replace `.skip()` with `.shuffle(seed)` in FinanceDataPipeline** — The previous approach called `dataset.skip(N)` on each streaming dataset, which downloads and discards thousands of items before yielding training examples. With `processed_items=67,720`, this took ~70 minutes per run — leaving only ~5 actual training steps in the 80-minute timeout window. New approach: each run derives a shuffle seed from the `processed_items` counter so different runs sample different dataset regions without the skip overhead. Training throughput: ~5 steps/run → ~120+ steps/run.

#### Critical — EWC State File Size (1.88 GB → ~158 MB)
- **Fix EWC pruning: top-K-ratio by Fisher magnitude** — The previous `FISHER_THRESHOLD` per-element check kept entire parameter tensors if ANY element exceeded the threshold — resulting in all 494M parameters being retained (1.88 GB file). New approach: keep top 8% of parameters by total Fisher magnitude (`FISHER_TOP_K_RATIO=0.08`). Result: ~39.5M params kept, ~158 MB EWC state file.

#### High — Broken Dataset
- **Replace `mhenrichsen/alpaca_data_cleaned` → `yahma/alpaca-cleaned`** — The `mhenrichsen/alpaca_data_cleaned` dataset was removed from HuggingFace Hub, causing a `[FAIL]` on every training run. `yahma/alpaca-cleaned` is the canonical maintained version of the same Alpaca cleaned dataset.

#### Medium — CI Fixes
- **Fix `delete_patterns` prefix in Upload Checkpoint step** — The `upload_folder(path_in_repo='checkpoint', delete_patterns=['checkpoint/pytorch_model.bin', ...])` was silently failing because huggingface_hub matches patterns relative to `path_in_repo`. Corrected to `['pytorch_model.bin', ...]`.
- **Add generation smoke test** — After each upload, run 2 finance prompts and log response quality (token count, uniqueness ratio). Catches silent generation failures before the next run builds on a broken checkpoint.

---

## [6.0.0] — 2026-05-26

> **Goal:** Fix factual accuracy failures, increase context window, reduce EWC overconstraint, shrink EWC state file from 1.88 GB to <200 MB, and improve per-run training throughput.

### Changes

#### Training Pipeline
- **BLOCK_SIZE: 256 → 512** — Doubles the context window per training sample. The current 256-token limit prevents the model from learning multi-step financial reasoning (e.g. DCF calculations, full option-pricing derivations).
- **MAX_BYTES: 15 MB → 25 MB** — More data per hourly run. At 15 MB the model processes ~240 examples per run; 25 MB raises this to ~400.
- **MAX_STEPS: 150 (unchanged)** — With BLOCK_SIZE doubling from 256 → 512, each step is ~2× more compute-intensive. Keeping MAX_STEPS at 150 maintains a comfortable margin within the 90-minute CI runner timeout (~60 min estimated).
- **EWC_LAMBDA: 500.0 → 75.0** — The current lambda=500 is overly conservative and has likely been slowing knowledge acquisition. Fisher estimates from only 5 samples at 500× strength overconstrain updates. Reducing to 75 balances retention with plasticity.
- **EWC_SAMPLES: 5 → 20** — The 5-sample Fisher estimate is too noisy. 20 samples gives a stable diagonal approximation without significant RAM overhead.
- **FISHER_THRESHOLD: 1e-4 → 5e-4** — The current threshold leaves the EWC state file at 1.88 GB (larger than the model itself at 942 MB). Raising the threshold to 5e-4 will prune ~80% of low-signal entries and bring EWC state down to ~200 MB.
- **GRAD_ACCUM: 8 → 4** — Reduces effective batch size, which can help with generalization on diverse finance tasks at the cost of slightly noisier gradients.

#### Dataset Curriculum
- **Add `yahma/alpaca-cleaned`** (weight 0.05) — Cleaned instruction-following data to improve response format consistency.
- **Reduce `nvidia/OpenMathInstruct-2` weight: 0.25 → 0.15** — Math instruction data is less critical than finance-specific factual data. The current 25% weight is too high given the factual errors observed.
- **Increase `sujet-ai/Sujet-Finance-Instruct-177k` weight: 0.12 → 0.18** — This is the highest-quality finance instruction dataset in the mix; increasing its share improves answer quality.

#### Architecture (Custom meridian/ module)
- **Fix NumeracyEncoder heuristic** — Current implementation uses `input_ids % 32` which is token-ID modulo, completely unrelated to actual numeric magnitude. New approach: detect digit tokens using the tokenizer vocabulary, assign magnitude buckets based on actual decoded numeric values.
- **Document architecture/training split clearly** — The custom `MeridianForCausalLM` is a research reference implementation, not what is trained in CI. This will be clearly marked in code, README, and docs.

#### Code Quality
- **Remove duplicate model.safetensors + pytorch_model.bin on HF** — Both files exist (totaling ~1.88 GB) because the CI `delete_patterns` fix was never applied retroactively. Cleanup script added.
- **Perplexity tracked per run** — Add validation perplexity logging to CI output and Comet ML so training progress is quantitatively visible.
- **Generation smoke test in CI** — After uploading, run 2 finance prompts and assert response length > 50 tokens to catch silent generation failures.

---

## [5.1.0] — 2026-02-26 to 2026-05-26 (3-month window)

> **Status: Active — 3 months of hourly training completed**

### Summary (Diagnostic — 2026-05-26)
- **Total items processed:** 64,464
- **Model perplexity on finance text:** 6.78 (excellent)
- **Generation quality (surface):** Good — coherent, on-topic responses
- **Generation quality (factual):** Needs improvement — specific factual errors detected (see Issues)

### What Worked
- Hourly CI pipeline ran reliably for 3 months without human intervention
- EWC (Elastic Weight Consolidation) successfully prevented catastrophic forgetting — early financial knowledge is retained
- Model produces coherent, formatted `### Instruction: / ### Response:` completions
- Perplexity of 6.78 on finance text indicates strong language modeling on finance domain
- Low repetition rate (0%) across all test prompts
- No NaN loss explosions or OOM failures in recent runs (RAM guard + soft throttle working)
- Dataset curriculum mix (25+ finance datasets) provides good topic diversity

### Issues Found

#### Critical — Factual Errors
- **Black-Scholes misattribution**: Model attributes the Black-Scholes model to "William Stanley J. Sharpe" rather than Fischer Black and Myron Scholes (1973). Sharpe created the Sharpe Ratio. This is a concrete factual error.
- **Compound interest arithmetic error**: For $10,000 at 5% over 10 years, the correct answer is $16,288.95. The model generates incorrect intermediate values, suggesting OpenMathInstruct training hasn't correctly transferred to finance math.
- **Dollar-cost averaging label confusion**: Model uses non-standard acronym "CDAC" for dollar-cost averaging, suggesting label noise in training data.

#### High — Architecture Mismatch
- **Custom SMoE never trained**: The `meridian/` module (MeridianForCausalLM with SMoE, GQA, RoPE, SwiGLU) exists in the codebase but is **never used** by `train.py`. The training pipeline fine-tunes `Qwen/Qwen2.5-0.5B` (standard Qwen2ForCausalLM). The README previously implied the custom arch was the trained model.
- **README / architecture.md now clarifies this**, but the headline features (SMoE, custom GQA) are attributes of the research module, not the deployed checkpoint.

#### High — EWC State File Size
- **EWC state: 1,884 MB** — The Fisher matrix state file is larger than the model weights (942 MB). With `FISHER_THRESHOLD=1e-4`, too many low-value entries are retained.
- **Impact**: GitHub Actions runners have 16 GB RAM. Loading a 1.88 GB EWC file + 942 MB model at training start consumes 2.82 GB before any training activations.

#### Medium — Duplicate Model Files on HuggingFace
- Both `model.safetensors` (942.3 MB) and `pytorch_model.bin` (942.4 MB) exist in the HuggingFace checkpoint. This wastes ~942 MB of storage on HF and causes confusion about which is canonical.
- Root cause: the CI `delete_patterns` flag for pytorch_model.bin was added but didn't remove the existing file.

#### Medium — Hyperparameter Constraints
- `BLOCK_SIZE=256` limits the model to seeing only 256 tokens of context during training. Finance documents (10-K filings, earnings reports, multi-step calculations) require longer context.
- `EWC_LAMBDA=500` with only `EWC_SAMPLES=5` gives a noisy Fisher estimate at very high regularization — may be slowing learning in recent months.

#### Low — Stale Script References
- `scripts/evaluate_model.py` references `HuggingFaceTB/SmolLM2-360M` as the base model comparison target (wrong — project migrated to Qwen2.5-0.5B).
- `scripts/hf_download_and_test.py` defaults to `hpcai-tech/openmoe-base` instead of `meridianal/FinAI`.
- `scripts/nuke_repo.py` references `MeridianAlgo/FinAI-Lite` (wrong repo).

#### Low — Repository Cleanliness
- `timing_test.py` (root-level debug script) committed to repo.
- `scripts/check_comet.py`, `scripts/find_workspace.py`, `scripts/check_cascade.py` contain hardcoded Comet API keys and are dev-only utilities.
- `.cometml-runs/` binary artifact tracked in git.
- `docs/examples/` duplicates the top-level `examples/` directory.

---

## [5.0.0] — 2025-12-01 (approx.)

### Changed
- **Base model upgraded: GPT-2 / SmolLM2-360M → Qwen/Qwen2.5-0.5B** — Significantly larger and more capable backbone. Qwen2.5 has stronger pre-training on code and math, making it a better foundation for finance+math reasoning.
- **Optimizer switched to AdaFactor** — AdamW's optimizer state (~2× params = ~1 GB) was exceeding GitHub Actions RAM limits. AdaFactor's factored second-moment approximation reduces optimizer RAM by ~80%.
- **EWC lambda increased: 100 → 500** — Increased conservatism to prevent catastrophic forgetting with the new larger model.
- **Tokenizer: GPT-2 (50,257 tokens) → Qwen2.5 (151,665 tokens)** — Much larger vocabulary improves sub-word representation, especially for numbers and finance terminology.
- **Architecture module updated**: `meridian/model/` custom SMoE updated to match Qwen2.5 tokenizer vocabulary size (151,665).

### Added
- Hard RAM guard (emergency save + exit at 14.5 GB)
- Soft RAM throttle (sequence truncation at 12.5 GB / 80%)
- `SKIP_OPTIMIZER_SAVE` flag to avoid saving 2 GB optimizer state to HuggingFace
- `FREE_OPTIMIZER_BEFORE_FISHER` flag to free optimizer RAM before Fisher computation

---

## [4.0.0] — 2025-10-01 (approx.)

### Changed
- **Base model: SmolLM2-135M → SmolLM2-360M** — Increased model capacity for better finance reasoning.
- **EWC added**: First integration of Elastic Weight Consolidation for continual learning.
- **Dataset curriculum expanded** to 25+ HuggingFace finance datasets.
- **CI: GitHub Actions hourly training** established and stabilized.

### Added
- `meridian/training/ewc.py` — Fisher Information Matrix computation and EWC penalty
- `sujet-ai/Sujet-Finance-Instruct-177k` dataset (12% weight)
- `FinGPT/fingpt-sentiment-train` dataset
- Weighted dataset mixing with round-robin iteration
- Gradient checkpointing for memory efficiency

---

## [3.0.0] — 2025-08-01 (approx.)

### Changed
- **Architecture redesign**: `meridian/model/` introduced with custom SMoE Transformer.
  - Sparse Mixture-of-Experts (8 experts, top-2 routing)
  - Grouped Query Attention (GQA)
  - RoPE position embeddings (theta=500,000)
  - SwiGLU feed-forward
  - RMSNorm
  - Financial Numeracy Encoding
- **Training split**: Custom `MeridianForCausalLM` used for smoke tests; `AutoModelForCausalLM` (HF pretrained) used for hourly CI training.
- **Load-balancing aux loss** (Switch Transformer style) added to MoE router.

### Added
- `meridian/model/configuration.py` — `MeridianConfig(PretrainedConfig)`
- `meridian/model/modeling.py` — Full Meridian SMoE implementation
- `SMOKE_TEST=1` mode for CI architecture verification
- Comet ML experiment tracking integration

---

## [2.0.0] — 2025-06-01 (approx.)

### Changed
- **Base model: DistilGPT-2 → SmolLM2-135M** — First move to a modern, instruction-following capable backbone.
- **Training data**: Expanded from 3 datasets to 10 finance-focused datasets.
- **Instruction format**: Standardized `### Instruction: / ### Response:` template.

### Added
- `meridian/data/pipeline.py` — `FinanceDataPipeline` streaming curriculum
- Shuffle-seed dataset sampling for continual training diversity (skip-ahead replaced in v6.0.1)
- `dataset_state.json` persisted to git for cross-run seed derivation
- NaN loss detection and batch skipping

---

## [0.1.0] — 2025-03-01 (approx.) — *prototype (was tagged 1.0.0)*

### Initial Prototype
- Proof-of-concept finance LLM with hourly CI training on GitHub Actions.
- Base model: DistilGPT-2
- Training data: `gbharti/finance-alpaca`, `HuggingFaceFW/fineweb-edu`, `FinanceMTEB/financial_phrasebank`
- Simple trainer: AdamW, no gradient accumulation, no EWC
- Basic checkpoint save/load via HuggingFace Hub
