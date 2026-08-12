# Phase 1 final results — 2026-08-12

Run [31617824586](https://github.com/MeridianAlgo/FinAI/actions/runs/31617824586). Complete sweep.
Supersedes `2026-08-11-sweep-1.md` (partial).

## Runner

`AMD EPYC 7763` (Zen 3), 4 vCPU, 15 GiB, torch 2.13.0, transformers 5.15.0.
`avx2` yes; `avx512f`, `avx512_bf16`, `amx_bf16` all **absent**; oneDNN available.

## Results

Llama-shaped decoder, vocab 16,384, block 512, fwd + bwd + AdamW step on synthetic tokens.

| Shape | Params | dtype | Batch | tok/s | GFLOP/s | Peak RSS |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 25M | 25.2M | fp32 | 1 | 519.5 | 78.5 | 1.3 GB |
| 25M | 25.2M | fp32 | 8 | **584.8** | 88.3 | 2.7 GB |
| 25M | 25.2M | fp32 | 32 | 652.1 | 98.5 | 4.7 GB |
| 57M | 56.6M | fp32 | 1 | 273.9 | 93.1 | 1.8 GB |
| 57M | 56.6M | fp32 | 8 | **304.9** | 103.6 | 3.9 GB |
| 57M | 56.6M | fp32 | 32 | 203.4 | 69.1 | 1.3 GB |
| 126M | 125.9M | fp32 | 1 | 135.9 | 102.6 | 3.0 GB |
| 126M | 125.9M | fp32 | 8 | **156.1** | 117.9 | 4.7 GB |
| 126M | 125.9M | fp32 | 32 | skipped | — | est. 14.0 GB |
| 494M-control | 494.0M | bf16 | 1 | **1.0** | 3.0 | 4.6 GB |

## 1. The control reproduces production exactly

The 494M/bf16/batch-1 control — today's `train.yml` configuration — measured **1.0 tok/s**
against the 0.92–1.00 tok/s logged in training run 31431031730.

That closes the diagnosis. Production throughput is not a data-pipeline problem, an EWC problem,
or a model-size problem. It is `DTYPE: 'bfloat16'` on hardware with no bf16 support, and nothing
else. The same 25M model runs at 519.5 tok/s in fp32 and 25.9 tok/s in bf16 — **20x**.

## 2. The runner is ~5x faster than the plan assumed

Measured fp32: **78–118 GFLOP/s**, against the 20 GFLOP/s `BASE_MODEL_PLAN.md` projected.
Efficiency *improves* with model size (88 -> 104 -> 118 GFLOP/s at batch 8) — larger matmuls use
the 4 cores better.

## 3. Batch 8 is the operating point, not batch 32

| Shape | b1 -> b8 | b8 -> b32 |
| --- | ---: | ---: |
| 25M | +13% | +12% |
| 57M | +11% | **−33%** |
| 126M | +15% | OOM (est. 14.0 GB) |

Batching buys ~1.3x, not the 3–6x estimated: at block 512 even batch 1 presents a 512-row GEMM,
so the cores are already near-saturated. Past batch 8 memory traffic dominates and 57M actually
*regresses*. **Batch 8** takes ~90% of peak throughput at ~40% of the memory.

## 4. Revised timelines

At batch 8 against ~1.17e6 training-seconds/month:

| Params | tok/s | Tokens/month | Chinchilla (20 tok/param) | Time to Chinchilla |
| ---: | ---: | ---: | ---: | ---: |
| 25M | 584.8 | 684 M | 504 M | **~3 weeks** |
| 57M | 304.9 | 357 M | 1.13 B | **~3.2 months** |
| 126M | 156.1 | 183 M | 2.52 B | ~13.8 months |

## Decision: start at 25M

25M reaches Chinchilla-optimal in ~3 weeks, which means a real model and a real perplexity curve
inside a month instead of a quarter. 57M is the better *final* model at 2.2x the capacity for 3.2
months, and remains reachable afterwards — either by training it fresh with everything learned
from the 25M run, or by continuing to over-train the 25M past Chinchilla, which is the right move
when inference cost matters.

Committing three months to 57M before the data pipeline has ever produced a descending loss curve
is the risk this ordering avoids.

## Method notes

- **Memory is predicted, not caught.** Configs above 10.5 GB estimated peak are skipped. Sweeps 2
  and 3 both died at 126M/batch32 with exit 143 — SIGTERM to the whole cgroup from systemd-oomd,
  which neither `except` nor subprocess isolation survives. The estimator classified all ten
  observed configs correctly, predicting 4.9 GB for 126M/batch8 against 4.7 GB measured.
- **`peak_rss_gb` is sampled after the step, not a true peak.** It reads low where a transient
  allocation dominated — 57M/batch32 reports 1.3 GB against an 8.8 GB estimate. Treat the
  estimates as the memory guide and RSS as corroboration.
- **Throughput is an upper bound.** Synthetic tokens mean zero data-loading cost, so the gap
  between these figures and the trainer's real rate is the pipeline's overhead.
