# Meridian Base Model Plan

Building **MeridianLM**, a from-scratch finance-native base model, instead of fine-tuning
Qwen2.5-0.5B.

Status: **Phase 1 in progress.**

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
against each other. Against a realistic post-fix 20 GFLOP/s:

```
Monthly budget:  466 CPU-hours x ~70% training = ~325 h = 1.17e6 s

N =  25 M  ->  133 tok/s  ->  156 M tokens/month
N =  60 M  ->   56 tok/s  ->   65 M tokens/month
N = 124 M  ->   27 tok/s  ->   31 M tokens/month
N = 494 M  ->    7 tok/s  ->    8 M tokens/month   (today, after fixes)

Chinchilla-optimal ~20 tokens/param:
  25 M params -> 500 M tokens -> ~3.2 months
  60 M params -> 1.2 B tokens -> ~18 months
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

### Phase 1 — Benchmark the box *(in progress)*

Every performance number above is an estimate until measured on a real runner. Sweep
`{25M, 57M, 126M} x {fp32, bf16} x {batch 1, 8, 32}` and record tok/s and GFLOP/s, plus a
494M/bf16/batch-1 control to reproduce today's 1 tok/s and confirm the diagnosis.

Primary hypothesis: **`DTYPE=bfloat16` is the main culprit.** Without AVX512-BF16 or AMX, PyTorch
emulates bf16 on CPU and it runs *slower* than fp32.

Deliverables: `scripts/benchmark_cpu.py`, `.github/workflows/benchmark.yml`, results committed to
`docs/benchmarks/`. Exit criterion: a measured tok/s figure for the 25M target that either
confirms or resizes the plan.

### Phase 2 — Corpus and tokenizer

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

| Change | Rationale | Est. |
| --- | --- | --- |
| fp32 instead of bf16 | Without AVX512-BF16/AMX, PyTorch emulates bf16 on CPU and it runs slower than fp32 | 2–4x |
| batch 32–64, not 1 | Batch 1 gives matrix-vector products with terrible arithmetic intensity | 3–6x |
| Pre-tokenized shards | Stops paying tokenization + streaming out of the training budget | 1.5–2x |
| Drop EWC | Fisher estimation costs a full pass per run to protect knowledge a from-scratch model lacks | 1.2x |
| AdamW over Adafactor | Adafactor trades convergence for optimizer memory we no longer need to save | quality |
| `torch.compile`, thread pinning | oneDNN fusion on the CPU backend | 1.3–2x |

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

- **Target size: 25M or smaller?** 25M/500M tokens is ~3 months; a 15M model reaches Chinchilla in
  ~6 weeks and proves the pipeline sooner. Recommendation: hold at 25M, let Phase 1 decide.
- **Keep the Qwen fine-tune running in parallel?** Costs nothing extra on free runners and
  preserves a fallback. Recommendation: keep it, reduced to every 6 hours.
- **Rent a GPU for pretraining?** ~$10–30 of spot A10/T4 covers what CI does in three months. If
  the goal is owning the model rather than the CPU constraint specifically, this is the highest-
  leverage question here.
