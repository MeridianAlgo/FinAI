# Phase 2 results — tokenizer and corpus, 2026-08-12

Final build: [31634323806](https://github.com/MeridianAlgo/FinAI/actions/runs/31634323806),
after adding SEC EDGAR and FinDB. Corpus at
[meridianal/FinAI-corpus](https://huggingface.co/datasets/meridianal/FinAI-corpus) (private).

## Deliverables

| Artifact | Value |
| --- | --- |
| Tokenizer | 16,384-vocab byte-level BPE, digits split individually |
| Training shards | **1,000,000,000 tokens**, uint16, 20 x 50M shards |
| Held-out validation | 2,000,000 finance + 2,000,000 general tokens |
| Domain mix | **69.97% finance / 30.03% general** (target 70%) |
| Unique source tokens | **889.9M finance**, 231.1M general |
| Repetition needed | **None** — 700M finance tokens drawn from 889.9M unique |

That is ~2x Chinchilla-optimal for the 25M target (504M tokens), so there is now room to
over-train, which is the main quality lever available for a model this size.

## Adding SEC EDGAR closed the data gap

The build before this one was finance-limited: 150.9M unique finance tokens forced ~2.2
epochs of repetition to reach even 500M tokens at 65% finance.

**EDGAR-CORPUS** (`c3po-ai/edgar-corpus`, the parquet mirror of Loukas et al. 2021) is
91,086 10-K filings from 1993–2020 — public domain, and it alone contributed **728,984,985
tokens** in 780s, a 5.9x increase in the unique finance pool.

Two things were needed to use it:

- It is published as a **loading script**, and `datasets` 4.x removed script execution, so
  `load_dataset` fails outright with "Dataset scripts are no longer supported". The Hub
  auto-converts every dataset to parquet on a `refs/convert/parquet` branch; those load
  natively, so `resolve_parquet_urls` asks the datasets-server for them (9 files, 2.19 GB).
- Filings arrive **split by item**, so `format_edgar` concatenates the substantive sections
  and drops any under 400 chars. An omitted item still emits its header plus "Not
  Applicable", and thousands of those would teach boilerplate rather than finance. Item 7
  (MD&A) carries the richest financial reasoning in a filing and is included.

**MeridianAlgo/FinDB** adds ~10M tokens of news prose, read from the SQLite file in the repo
rather than the Hub. Half of it is unusable, though — see below.

### FinDB data quality

| Source | Articles | Avg chars | Usable |
| --- | ---: | ---: | ---: |
| google_finance | 11,387 | 398 | **0%** |
| cnbc | 2,709 | 4,368 | 100% |
| guardian_business | 2,149 | 4,995 | 96% |
| yahoo_finance | 3,519 | 3,403 | 95% |
| bbc_business | 971 | 4,015 | 94% |
| marketwatch | 989 | 526 | 82% |
| seeking_alpha | 1,098 | 325 | 34% |

`google_finance` — half the database by article count — stores unparsed Google News redirect
URLs instead of article text, e.g.
`a hrefhttps:news.google.comrssarticlesCBMi9gFBVV95cUxQeTNRU1BK...`. **This looks like a
scraper bug in FinDB worth fixing at the source**, since those rows also carry sentiment
scores and entity extractions computed over a URL. Excluded here, along with a prose
heuristic and U+FFFD stripping (mojibake from an upstream mis-decode), leaving 9,839 clean
articles of 22,822 at 4,056 chars average.

## Earlier build (superseded)

The 500M-token build was: 64.96% finance, 150.9M unique finance tokens, ~2.2 epochs of
repetition.

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

- ~~The corpus is finance-limited~~ — resolved by EDGAR. 889.9M unique finance tokens now
  support a 1B-token corpus with no repetition, at ~2x Chinchilla for the 25M target.
- **Vocab 8,192 vs 16,384** is unresolved; 8k is ~9% cheaper per unit of text.
- **FinDB's google_finance rows are broken at the source.** Excluded here, but half that
  database is storing redirect URLs where article text should be.
- **Licensing.** The shards are a derivative of the source datasets and reconstructible to
  text given the tokenizer, so the HF repo defaults to private. Making it public is a
  deliberate call, not a build flag.
