# Phase 2 results — tokenizer and corpus, 2026-08-12

Final build: [31628078595](https://github.com/MeridianAlgo/FinAI/actions/runs/31628078595).
Corpus at [meridianal/FinAI-corpus](https://huggingface.co/datasets/meridianal/FinAI-corpus)
(private).

## Deliverables

| Artifact | Value |
| --- | --- |
| Tokenizer | 16,384-vocab byte-level BPE, digits split individually |
| Training shards | 500,000,000 tokens, uint16, 10 x 50M shards |
| Held-out validation | 2,000,000 finance + 2,000,000 general tokens |
| Domain mix | **64.96% finance / 35.04% general** (target 65%) |
| Unique source tokens | 150.9M finance, 139.5M general |

## 1. Fourteen of twenty-seven datasets were producing nothing

The first build revealed that 14 configured datasets yielded **zero documents**. Their spec
named a column the dataset does not have, so `_format_text` returned `""` and every row was
dropped without a word of complaint.

`_format_text` is shared with the trainer, so **this was equally true of live training** —
the real mix has been roughly half its documented composition for as long as those entries
have existed. Three causes:

| Cause | Sets | Fix |
| --- | --- | --- |
| Column is `sentence`, not `text` | 6 | `text_field: sentence`, `label_field: label_text` |
| MTEB retrieval set: `default` config holds only qrels | 4 | `config: corpus` (FinQA, TATQA, TradeTheEvent x2) |
| Ships only a `test` split | 1 | `split: test` |
| Chinese-language, or no readable schema | 4 | dropped |

Recovering FinQA and TATQA matters disproportionately — TATQA's corpus is markdown tables of
financial statements, which is close to ideal pretraining material for this model.

`scripts/validate_datasets.py` now checks every spec against the live Hub schema and exits
non-zero on any break; `corpus.yml` gates on it. **23/23 pass, covering 100% of mix weight.**

## 2. The tokenizer claim in the plan was wrong

The plan asserted a 16k domain tokenizer would cut token count 10–20% versus Qwen. Measured
on held-out finance probes, trained on 277,644 sampled documents:

| Vocab | Tokens | chars/token | Model params @ d384 | Relative cost |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 340 | 3.16 | 22.0M | **7.5** |
| 16,384 | 324 | 3.31 | 25.2M | 8.2 |
| 32,768 | 312 | 3.44 | 31.5M | 9.8 |
| Qwen 151,643 | 311 | 3.45 | 77.1M | 24.0 |

Our tokenizer produces **more** tokens than Qwen's at every size tested, not fewer. A 16k
vocab cannot hold as many whole English words — ours splits `Adjusted` into `Ad`/`just`/`ed`
where Qwen has it whole — and that loss outweighs the win on jargon (`EBITDA` in 2 tokens
against Qwen's 3).

Compression was the wrong objective. Training compute is `6 x params x tokens`, and vocab
size trades embedding parameters against token count, so the quantity to minimize is their
**product**. On that measure the domain tokenizer wins overwhelmingly: **68.8% cheaper than
Qwen's vocab** at the same d_model.

Two things worth noting. At 32,768 our tokenizer reaches 3.44 chars/token against Qwen's
3.45 — **matching its compression with a vocab 4.6x smaller**, which is the domain advantage
properly stated. And the cost-optimal choice is 8,192, ~9% cheaper than the 16,384 we kept;
16k was retained for context efficiency, since a larger vocab fits more text into the same
512-token window. That remains an open call.

## 3. The finance/general ratio does not fall out of weighting

Two failures before this landed:

**Build 1 came out 70% general** for a finance model, and reached only 284M of 500M
requested. The finance datasets are finite and small; fineweb-edu and OpenMathInstruct are
effectively unlimited. Once finance ran dry, its share was redistributed to whoever was
still live. Fixed by taking finance to exhaustion, sizing the general pull to the target
ratio, and adding bounded repetition (≤4 epochs — standard practice for scarce data, and
close to as useful as fresh tokens).

**Build 2 came out 46.5% finance** against a 65% target. Sampling over one flat pool still
leaked: as small finance sets hit the epoch cap and retired, `rng.choices` renormalized
across *both* domains and general absorbed the slack. Fixed by choosing the domain first and
only then a source within it, which makes the ratio independent of which sources survive.

Final: **64.96%**.

## Open items for Phase 3

- **The corpus is finance-limited, not compute-limited.** 500M tokens needs 325M finance
  tokens, met only by repeating 150.9M unique tokens ~2.2x. Chinchilla for 25M is 504M
  tokens, so this is exactly at budget with no room to over-train. More unique finance text
  is the single highest-value addition — **SEC EDGAR filings are public domain** and absent
  from the dataset list entirely, despite the plan assuming ~30% of the mix from them.
- **Vocab 8,192 vs 16,384** is unresolved; 8k is ~9% cheaper per unit of text.
- **Licensing.** The shards are a derivative of the source datasets and reconstructible to
  text given the tokenizer, so the HF repo defaults to private. Making it public is a
  deliberate call, not a build flag.
