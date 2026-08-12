"""Phase 2: train the 16k finance BPE and measure it against Qwen's.

Why a domain tokenizer is the highest-leverage change in the plan, per
``docs/BASE_MODEL_PLAN.md``:

1. Parameters. Qwen's 152k vocab at d_model 384 is a 58M-parameter embedding table — more
   than twice the entire 25M target model. At 16k it is 6.3M.
2. Compression. General tokenizers shatter finance text: ``EBITDA``, ``10-Q``, ticker
   symbols, ``$1.4B``. Fewer tokens for the same content is a straight multiplier on
   effective data, and this script measures that rather than assuming it.

Two deliberate choices:

* **Digits are split individually.** Llama does this and it measurably helps arithmetic:
  a model that sees ``1``/``4``/``.``/``2`` learns place value, whereas one that sees a
  single ``14.2`` token has to memorize each magnitude separately. For a model expected to
  reason about financial figures that trade is worth the extra tokens.
* **Byte-level BPE**, so there is no UNK token and any input is representable.

Usage:
    python scripts/build_tokenizer.py --docs-per-dataset 20000 --vocab-size 16384
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterator

from meridian.data.corpus import dataset_specs, iter_documents

EOS_TOKEN = "<|endoftext|>"

# Held-out probes for the compression comparison. Deliberately the kind of text the model
# is meant to be good at: filing prose, figures, tickers, and finance jargon.
PROBE_TEXTS = [
    "Total revenue increased 14.2% to $1.43 billion for the three months ended June 30, 2026, "
    "driven by growth in subscription services.",
    "Adjusted EBITDA margin contracted 210 basis points year-over-year to 23.4%, reflecting "
    "elevated SG&A expense and unfavorable FX translation.",
    "The Company's 10-Q filed with the SEC discloses a $47.5 million goodwill impairment "
    "charge related to the EMEA reporting unit.",
    "NVDA closed at $184.72, up 2.3%, while the S&P 500 gained 0.8% and the CBOE VIX fell to "
    "13.6 on lighter-than-average volume.",
    "Free cash flow conversion of 87% and a net leverage ratio of 2.1x EBITDA support the "
    "board's decision to authorize a $500 million buyback.",
    "Diluted earnings per share of $2.14 missed consensus estimates of $2.21, and management "
    "guided FY27 revenue to $6.2-6.4 billion.",
]


def sample_corpus(
    docs_per_dataset: int, include_heavy: bool, max_chars_per_dataset: int
) -> Iterator[str]:
    """Yield training text for the BPE, visiting one dataset at a time."""
    specs = dataset_specs(include_heavy=include_heavy)
    print(f"  Sampling from {len(specs)} datasets\n", flush=True)

    for spec in specs:
        # Allocate documents by mix weight, with a floor so small datasets still contribute
        # their vocabulary (FOMC language, ESG terms) rather than being rounded away.
        quota = max(200, int(docs_per_dataset * spec["weight"] * len(specs)))
        quota = min(quota, docs_per_dataset)
        count = 0
        started = time.perf_counter()
        for text in iter_documents(spec, EOS_TOKEN, quota, max_chars_per_dataset):
            count += 1
            yield text
        print(
            f"    {spec['name']:<48} {count:>7,} docs  "
            f"({spec['domain']}, {time.perf_counter() - started:.0f}s)",
            flush=True,
        )


def build(args: argparse.Namespace) -> dict:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # Digits first so numbers are split before byte-level grouping can fuse them.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=2,
    )

    print("=" * 78)
    print("  Phase 2 — training finance BPE")
    print("=" * 78)
    started = time.perf_counter()
    tokenizer.train_from_iterator(
        sample_corpus(args.docs_per_dataset, args.include_heavy, args.max_chars_per_dataset),
        trainer=trainer,
    )
    elapsed = time.perf_counter() - started
    print(f"\n  Trained in {elapsed:.0f}s, vocab {tokenizer.get_vocab_size():,}\n", flush=True)

    os.makedirs(args.out, exist_ok=True)
    tokenizer.save(os.path.join(args.out, "tokenizer.json"))

    # Wrap as a PreTrainedTokenizerFast so the rest of the stack -- from_pretrained, the
    # trainer, generate(), the HF upload -- treats it like any other tokenizer.
    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(args.out, "tokenizer.json"),
        eos_token=EOS_TOKEN,
        bos_token=None,
        unk_token=None,
        pad_token=EOS_TOKEN,
    )
    fast.save_pretrained(args.out)
    print(f"  Saved tokenizer to {args.out}/", flush=True)
    return {"vocab_size": tokenizer.get_vocab_size(), "train_seconds": round(elapsed, 1)}


def compare(args: argparse.Namespace) -> dict:
    """Measure compression against Qwen's tokenizer on held-out finance text."""
    from transformers import AutoTokenizer

    ours = AutoTokenizer.from_pretrained(args.out)
    try:
        theirs = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as exc:  # noqa: BLE001 — the comparison is informative, not required
        print(f"  [WARN] Could not load Qwen tokenizer for comparison: {exc}")
        return {}

    total_chars = sum(len(t) for t in PROBE_TEXTS)
    ours_tokens = sum(len(ours.encode(t)) for t in PROBE_TEXTS)
    theirs_tokens = sum(len(theirs.encode(t)) for t in PROBE_TEXTS)

    print("  Compression on held-out finance text")
    print("  " + "-" * 62)
    print(f"    {'':<22}{'tokens':>10}{'chars/token':>14}{'vs Qwen':>12}")
    print(
        f"    {'Meridian 16k':<22}{ours_tokens:>10,}{total_chars / ours_tokens:>14.2f}"
        f"{(theirs_tokens - ours_tokens) / theirs_tokens * 100:>11.1f}%"
    )
    print(f"    {'Qwen2.5 152k':<22}{theirs_tokens:>10,}{total_chars / theirs_tokens:>14.2f}")
    print()

    for text in PROBE_TEXTS[:2]:
        print(f"    {text[:70]}...")
        print(f"      ours ({len(ours.encode(text)):>3}): {ours.tokenize(text)[:18]}")
        print(f"      qwen ({len(theirs.encode(text)):>3}): {theirs.tokenize(text)[:18]}")
    print()

    embed_ours = ours.vocab_size * 384
    embed_theirs = theirs.vocab_size * 384
    print(f"  Embedding table at d_model 384: {embed_ours / 1e6:.1f}M vs {embed_theirs / 1e6:.1f}M")
    print(f"  Saved: {(embed_theirs - embed_ours) / 1e6:.1f}M parameters\n")

    return {
        "probe_chars": total_chars,
        "meridian_tokens": ours_tokens,
        "qwen_tokens": theirs_tokens,
        "token_reduction_pct": round((theirs_tokens - ours_tokens) / theirs_tokens * 100, 2),
        "meridian_chars_per_token": round(total_chars / ours_tokens, 3),
        "qwen_chars_per_token": round(total_chars / theirs_tokens, 3),
        "embedding_params_saved": embed_theirs - embed_ours,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--docs-per-dataset", type=int, default=20000)
    parser.add_argument("--max-chars-per-dataset", type=int, default=60_000_000)
    parser.add_argument("--include-heavy", action="store_true", default=True)
    parser.add_argument("--no-include-heavy", dest="include_heavy", action="store_false")
    parser.add_argument("--out", default="tokenizer")
    parser.add_argument("--report", default="docs/benchmarks/tokenizer.json")
    args = parser.parse_args()

    report = build(args)
    report.update(compare(args))

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Wrote {args.report}")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path and report.get("qwen_tokens"):
        with open(summary_path, "a") as fh:
            fh.write("## Phase 2 — finance tokenizer\n\n")
            fh.write(
                f"Vocab **{report['vocab_size']:,}**, trained in {report['train_seconds']}s\n\n"
            )
            fh.write("| Tokenizer | Tokens on probe set | chars/token |\n| --- | ---: | ---: |\n")
            fh.write(
                f"| Meridian 16k | {report['meridian_tokens']:,} | "
                f"{report['meridian_chars_per_token']} |\n"
            )
            fh.write(
                f"| Qwen2.5 152k | {report['qwen_tokens']:,} | "
                f"{report['qwen_chars_per_token']} |\n\n"
            )
            fh.write(f"**{report['token_reduction_pct']}% fewer tokens** on finance text, ")
            fh.write(f"and {report['embedding_params_saved'] / 1e6:.1f}M fewer embedding params.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
