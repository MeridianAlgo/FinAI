# Meridian Base Model Plan

Building **MeridianLM**, a from-scratch finance-native base model, instead of fine-tuning
Qwen2.5-0.5B.

Status: **Phases 1 and 2 complete.** Phase 3 (dense 25M baseline) is next.

- Phase 1: `docs/benchmarks/2026-08-12-sweep-final.md`
- Phase 2: `docs/benchmarks/2026-08-12-phase2-corpus.md`

Phase 1 confirmed the diagnosis and moved three numbers materially:

- **`DTYPE: 'bfloat16'` is the whole of the 1 tok/s problem.** The 494M control reproduced
  production at exactly 1.0 tok/s, and fp32 is **20x faster** than bf16 on this hardware.
- **The runner does 78–118 GFLOP/s, not the ~20 assumed** — so 25M reaches Chinchilla-optimal in
  **~3 weeks**, not 3.2 months.
- **Batch 8, not 32.** Batching buys ~1.3x, not 3–6x, and 57M *regresses* 33% at batch 32.

---

## Why

Telemetry from training run `31431031730` (2026-08-10):

| Measure | Value | Source |
| --- | --- | --- |
| Throughput | **1.0 tok/s** | `tokens_per_sec` in Comet log |
| Optimizer steps per 80-min run | 19 | global step 37,336 → 37,355 |
| Tokens per run | ~4,800 | 19 steps x batch 1 x accum 2 x block 256 |
| Cumulative tokens, all time | ~19.1 M | 37,336 steps x 512 tok |
| Qwen2.5-0.5B pretraining budget | 18 T | Qwen tech report |
| Our share of that | **0.0001 %** | 19.1M / 18T |
| Loss range within one run | 0.78 – 2.98 | Comet `loss [24]` |

At 1 tok/s on a 494M-parameter model we extract ~3 GFLOP/s from a runner that should deliver
30–80 — roughly 4–10% of the hardware. And 19M cumulative tokens against an 18T-token base makes
the fine-tune statistically invisible; with EWC pinning the weights, the loss swinging 0.78–2.98
inside a single run is sampling noise, not learning.

We are spending ~470 CPU-hours/month for approximately no model change. That is what makes a
from-scratch model worth doing: the compute is already being paid for.

## The constraint

Training compute is about `6 x params x tokens` FLOPs, so parameters and tokens trade directly
against each other. **Measured** in Phase 1, fp32 at batch 8:

```
Monthly budget:  466 CPU-hours x ~70% training = ~325 h = 1.17e6 s

N =  25 M  ->  584.8 tok/s  ->  684 M tokens/month  -> Chinchilla (504 M) in ~3 weeks
N =  57 M  ->  304.9 tok/s  ->  357 M tokens/month  -> Chinchilla (1.13 B) in ~3.2 months
N = 126 M  ->  156.1 tok/s  ->  183 M tokens/month  -> Chinchilla (2.52 B) in ~13.8 months
N = 494 M  ->    1.0 tok/s  (bf16, today's config — the control that reproduced production)
```

**"Really smart in general" and "trained from scratch on CI CPUs" cannot both be true.** General
capability tracks total training FLOPs. Frontier models use ~1e24; our annual budget is ~1e17.
Seven orders of magnitude — no architecture recovers that.

What is achievable: at ~25M parameters on a narrow, curated corpus, models become genuinely
fluent (the TinyStories result). The target is a model that is **fluent and accurate in finance
and openly bad at everything else** — and that we own end to end.

## Target architecture — MeridianLM-25M

Llama-shaped decoder, deliberately conventional in the backbone so existing checkpoint
save/load, HF upload, `generate()`, and the model card keep working unchanged.
`build_smoke_model()` in `meridian/model/__init__.py` is already this pattern.

| Component | Value | Rationale |
| --- | --- | --- |
| Vocab | 16,384 | Finance-trained BPE — biggest single lever |
| d_model | 384 | Wide enough for efficient CPU GEMMs |
| Layers | 12 | Depth without serial-latency blowup |
| Heads / KV heads | 6 / 2 | GQA, 3x smaller KV cache at inference |
| FFN intermediate | 1,024 | SwiGLU (3 matrices at 2.67x d_model) |
| Context | 512 | RoPE, extensible later |
| Norm | RMSNorm | Cheaper than LayerNorm, pre-norm |
| Embeddings | tied | Saves 6.3M params |
| **Total** | **25.2 M** | 18.9M non-embedding |

### Efficiency thesis

The bottleneck is **FLOPs per token, not memory** — 25M params plus AdamW states is ~300 MB of a
16 GB box. That asymmetry points at mixture-of-experts, which buys capacity in memory rather than
compute:

```
Phase 4 MoE variant: 8 experts, top-1 routing, FFN only
  total params   25 M -> 124 M    (8x the FFN weights)
  active params  25 M ->  25 M    (top-1 routes to one expert)
  FLOPs/token    unchanged
  RAM            0.3 GB -> 1.5 GB (of 16 GB)
```

5x the capacity in the one currency we have a surplus of. Also the riskiest part — top-1 routing
needs a load-balancing loss to avoid expert collapse, and gather/scatter is weaker on CPU — so it
ships only if it beats the dense baseline at equal token count.

---

## Phases

### Phase 1 — Benchmark the box *(complete)*

Hypothesis confirmed: `DTYPE=bfloat16` on a CPU without AVX512-BF16 or AMX is emulated in
software and costs 20x. The 494M control landed at exactly 1.0 tok/s, matching production.

Full results and method notes in `docs/benchmarks/2026-08-12-sweep-final.md`. Deliverables:
`scripts/benchmark_cpu.py`, `.github/workflows/benchmark.yml`.

**Target size decision: start at 25M.** It reaches Chinchilla-optimal in ~3 weeks, so the data
pipeline gets a real perplexity curve inside a month. 57M is the better final model — 2.2x the
capacity for 3.2 months — and stays reachable afterwards, either trained fresh with what the 25M
run teaches or by over-training the 25M past Chinchilla. Committing a quarter to 57M before the
pipeline has ever produced a descending loss curve is the risk this ordering avoids.

### Phase 2 — Corpus and tokenizer *(complete)*

Results in `docs/benchmarks/2026-08-12-phase2-corpus.md`. Delivered: a 16,384-vocab finance
BPE, **1B training tokens** in 20 uint16 shards at **69.97% finance**, and 2M held-out tokens
per domain, at [meridianal/FinAI-corpus](https://huggingface.co/datasets/meridianal/FinAI-corpus).
That is ~2x Chinchilla for the 25M target, so over-training is on the table.

Findings that revise this plan:

- **14 of 27 datasets were yielding zero documents** — their spec named a column that does
  not exist, so `_format_text` returned `""` and every row was dropped silently. Since the
  trainer shares `_format_text`, live training has had the same hole. Now fixed and gated by
  `scripts/validate_datasets.py`; 25/25 specs pass.
- **The tokenizer compression claim below was wrong.** Ours produces *more* tokens than
  Qwen's, not 10–20% fewer. It still wins decisively on the metric that matters —
  `params x tokens`, i.e. training cost for a fixed body of text — by **66%**.
- **SEC EDGAR closed the data gap.** The finance pool was 150.9M unique tokens, needing ~2.2
  epochs of repetition; EDGAR-CORPUS added **729M tokens** of public-domain 10-K filings,
  taking it to 889.9M and eliminating repetition entirely.
- **FinDB contributes ~10M tokens of news prose**, but half that database — every
  `google_finance` row — stores unparsed redirect URLs instead of article text. Excluded
  here; worth fixing in the FinDB scraper.

Original plan below, kept for context.

### Phase 2 — Corpus and tokenizer (as planned)

Assemble the mix, train a 16k finance BPE, measure its compression against Qwen's on held-out
filings, pre-tokenize to `uint16` shards, publish as an HF dataset repo.

Corpus target: ~30% SEC EDGAR filings (10-K/10-Q/8-K, public domain), ~10% finance QA (FinQA,
ConvFinQA), ~15% market prose (news, earnings calls), ~45% general (FineWeb-Edu sample, Simple
Wikipedia). General text is scaffolding — a model that has never seen ordinary English cannot
write a coherent sentence about a balance sheet.

Two wins from the tokenizer: the embedding table drops from 58M (152k x 384) to 6.3M, and a
domain BPE should cut token count 10–20% on finance text, a straight multiplier on effective data.

**Pre-tokenized shards, not live streaming.** The current pipeline streams ~25 HF datasets
concurrently and tokenizes on the fly — per the comments in `train.yml`, that is what drove RSS
past 15 GB, forced `SHUFFLE_BUFFER` to 128, and made `fineweb-edu` unusable. Tokenize once, write
`uint16` shards, `np.memmap` at train time: constant memory, near-zero CPU overhead, reproducible
ordering, and the RAM guards become vestigial.

### Phase 3 — Dense 25M baseline

Config-driven `build_meridian_model()`, memmap dataloader, AdamW, cosine schedule with warmup,
held-out perplexity per run. Train to 500M tokens (~3 months). This is the model that either
works or does not; everything after is optimization.

### Phase 4 — MoE variant, gated on evidence

8-expert top-1 FFN with load-balancing loss, same corpus, compared at equal token count. Ships
only if held-out perplexity improves.

### Phase 5 — Instruction tuning

Supervised fine-tune on finance instruction pairs. This is where EWC finally earns its place —
protecting pretraining knowledge is exactly its job, and unlike now there will be something worth
protecting.

---

## Throughput fixes (Phase 1 findings feed these)

Independent of architecture, worth doing regardless. Multipliers are hypotheses to test, in the
order to test them.

| Change | Rationale | Est. | Measured |
| --- | --- | --- | --- |
| fp32 instead of bf16 | No AVX512-BF16/AMX on the runner, so PyTorch emulates bf16 in software | 2–4x | **20x** |
| batch 8, not 1 | Better arithmetic intensity — but block 512 already gives a 512-row GEMM | 3–6x | **1.3x** (and batch 32 *regresses* at 57M) |
| Pre-tokenized shards | Stops paying tokenization + streaming out of the training budget | 1.5–2x | not yet measured |
| Drop EWC | Fisher estimation costs a full pass per run to protect knowledge a from-scratch model lacks | 1.2x | not yet measured |
| AdamW over Adafactor | Adafactor trades convergence for optimizer memory we no longer need to save | quality | — |
| `torch.compile`, thread pinning | oneDNN fusion on the CPU backend | 1.3–2x | not yet measured |

**Scheduling.** GitHub-hosted jobs may run up to 6 hours; we do 80-minute jobs hourly. Each run
pays ~10 min of setup, checkpoint pull, and upload — a ~20% tax plus 4x the Hub traffic that keeps
triggering 429 handling. A 5.5-hour job every 6 hours cuts the tax to ~4%.

## Evaluation

The current smoke test checks that generated text has a unique-token ratio above 0.4. That passes
for fluent nonsense and for a model that has not moved in a month — it told us nothing about the
1 tok/s problem. Replace with:

- **Held-out perplexity every run** on two frozen splits, finance and general, never trained on.
  The only signal that distinguishes learning from noise, and the gate for promoting a checkpoint.
- **A fixed 200-prompt finance probe set**, exact-match on numeric answers and terminology, nightly.
- **A tok/s floor in CI.** A regression check at 50 tok/s would have surfaced this in July.

## Risks

- **Actions Terms of Service.** GitHub restricts Actions to building, testing, and publishing the
  project. Continuous model training on free public-repo runners is a grey area; near-100% duty
  cycle for months makes it conspicuous, and the worst case is org-wide Actions restriction with
  little warning. Worth pricing a small spot GPU — $10–30 would compress three months of this into
  about a day.
- **Three-month feedback loop.** Mitigation: perplexity should visibly descend within the first
  week, and a 5M-param dry run over ~20M tokens validates the pipeline end to end in a couple of days.
- **The result will be narrow.** MeridianLM-25M will produce confident nonsense outside finance.
  That belongs in the model card, not in a user's bug report.
- **Sunk cost.** This plan discards global step 37,336 of Qwen fine-tuning. Per the table above,
  there is functionally nothing there to lose.

## Open decisions

- ~~**Target size**~~ — settled by Phase 1: start at 25M (~3 weeks to Chinchilla), revisit 57M after.
- **Set `DTYPE: 'float32'` in `train.yml` now?** The running Qwen fine-tune is losing 20x to bf16
  emulation today. The catch is that fp32 doubles weight and gradient memory on a 494M model
  against a pipeline whose RAM guards (`MAX_RAM_GB: 14.5`) were tuned for bf16, so it needs one
  supervised run to confirm it fits. Worth doing only if that pipeline is staying alive.
- **Keep the Qwen fine-tune running in parallel?** Costs nothing extra on free runners and
  preserves a fallback. Recommendation: keep it, reduced to every 6 hours.
- **Rent a GPU for pretraining?** ~$10–30 of spot A10/T4 covers what CI does in three months. If
  the goal is owning the model rather than the CPU constraint specifically, this is the highest-
  leverage question here.
