"""Phase 2: train the finance BPE and pick its vocab size from measurement.

The plan asserted a 16k domain tokenizer would cut token count 10-20% against Qwen's 152k.
The first build measured the opposite: 250 tokens vs 233, i.e. **7.3% worse**. A 16k vocab
simply cannot hold as many whole English words -- ours splits ``Adjusted`` into
``Ad``/``just``/``ed`` where Qwen has it whole -- and that loss outweighs the win on finance
jargon (``EBITDA`` in 2 tokens against Qwen's 3).

So compression alone is the wrong objective. What actually matters for a fixed compute
budget is how much text a model can process per FLOP, and training compute is
``6 x params x tokens``. A bigger vocab means more embedding parameters but fewer tokens for
the same text, so the quantity to minimize is their **product**:

    cost = (embedding_params + backbone_params) x tokens_for_a_fixed_corpus

This script sweeps vocab sizes and reports that product, which turns vocab size into a
measurement instead of a guess. The parameter side is not small: at d_model 384, Qwen's 152k
vocab would be a 58.2M embedding table against a 18.9M backbone.

Two other deliberate choices:

* **Digits are split individually**, so the model learns place value rather than memorizing
  each magnitude. Qwen does the same, so this costs nothing in the comparison.
* **Byte-level BPE**, so there is no UNK and any input is representable.

Usage:
    python scripts/build_tokenizer.py --sweep 8192 16384 32768 --vocab-size 16384
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterator

from meridian.data.corpus import dataset_specs, iter_documents

EOS_TOKEN = "<|endoftext|>"

# Non-embedding parameters of the 25M target from docs/BASE_MODEL_PLAN.md. Vocab size does
# not change the backbone, so it is the fixed term when comparing total model cost.
BACKBONE_PARAMS = 18_883_584
D_MODEL = 384

# Held-out probes for the compression comparison: filing prose, figures, tickers, jargon.
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
    "Net cash provided by operating activities was $1.28 billion, compared with $974 million "
    "in the prior-year period, primarily due to favorable working capital.",
    "The FOMC held the federal funds target range at 4.25-4.50% and signaled two cuts in 2027 "
    "as core PCE inflation moderated to 2.4%.",
]


def sample_to_file(path: str, docs_per_dataset: int, include_heavy: bool, max_chars: int) -> int:
    """Materialize the BPE training sample once, as JSONL.

    Written to disk rather than held in memory so the same sample can train every vocab size
    in the sweep -- streaming the corpus once per size would be both slow and, because the
    sample would differ, not a controlled comparison.
    """
    specs = dataset_specs(include_heavy=include_heavy)
    print(f"  Sampling from {len(specs)} datasets\n", flush=True)
    total = 0

    with open(path, "w", encoding="utf-8") as fh:
        for spec in specs:
            # Allocate by mix weight with a floor, so small sets still contribute their
            # vocabulary (FOMC language, ESG terms) instead of being rounded away.
            quota = max(500, int(docs_per_dataset * spec["weight"] * len(specs)))
            quota = min(quota, docs_per_dataset)
            count = 0
            started = time.perf_counter()
            for text in iter_documents(spec, EOS_TOKEN, quota, max_chars):
                fh.write(json.dumps(text) + "\n")
                count += 1
            total += count
            status = "" if count else "   <-- YIELDED NOTHING"
            print(
                f"    {spec['name']:<48} {count:>7,} docs  "
                f"({spec['domain']}, {time.perf_counter() - started:.0f}s){status}",
                flush=True,
            )

    print(f"\n  Sampled {total:,} documents to {path}\n", flush=True)
    return total


def read_sample(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def train_bpe(vocab_size: int, sample_path: str):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # Digits first, so numbers are split before byte-level grouping can fuse them.
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
        min_frequency=2,
    )
    started = time.perf_counter()
    tokenizer.train_from_iterator(read_sample(sample_path), trainer=trainer)
    return tokenizer, time.perf_counter() - started


def measure(encode, vocab_size: int) -> dict:
    """Compression and total-model-cost metrics for one tokenizer."""
    chars = sum(len(t) for t in PROBE_TEXTS)
    tokens = sum(len(encode(t)) for t in PROBE_TEXTS)
    embedding = vocab_size * D_MODEL
    total_params = embedding + BACKBONE_PARAMS
    return {
        "vocab_size": vocab_size,
        "tokens": tokens,
        "chars_per_token": round(chars / tokens, 3),
        "embedding_params": embedding,
        "total_params": total_params,
        # Proportional to the FLOPs needed to train on a fixed body of text.
        "relative_cost": total_params * tokens,
    }


def save(tokenizer, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

    # Wrap as PreTrainedTokenizerFast so from_pretrained, the trainer, generate(), and the
    # HF upload all treat it like any other tokenizer.
    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(out_dir, "tokenizer.json"),
        eos_token=EOS_TOKEN,
        bos_token=None,
        unk_token=None,
        pad_token=EOS_TOKEN,
    )
    fast.save_pretrained(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=16384, help="Vocab size to keep")
    parser.add_argument("--sweep", type=int, nargs="*", default=[8192, 16384, 32768])
    parser.add_argument("--docs-per-dataset", type=int, default=40000)
    parser.add_argument("--max-chars-per-dataset", type=int, default=120_000_000)
    parser.add_argument("--include-heavy", action="store_true", default=True)
    parser.add_argument("--no-include-heavy", dest="include_heavy", action="store_false")
    parser.add_argument("--out", default="tokenizer")
    parser.add_argument("--sample-file", default="tokenizer_sample.jsonl")
    parser.add_argument("--report", default="docs/benchmarks/tokenizer.json")
    args = parser.parse_args()

    print("=" * 78)
    print("  Phase 2 — finance tokenizer")
    print("=" * 78)
    documents = sample_to_file(
        args.sample_file, args.docs_per_dataset, args.include_heavy, args.max_chars_per_dataset
    )
    if documents == 0:
        raise SystemExit("FATAL: sampled zero documents")

    sizes = sorted({*(args.sweep or []), args.vocab_size})
    rows, kept = [], None
    for size in sizes:
        tokenizer, seconds = train_bpe(size, args.sample_file)
        row = measure(lambda t: tokenizer.encode(t).ids, tokenizer.get_vocab_size())
        row["train_seconds"] = round(seconds, 1)
        rows.append(row)
        print(
            f"  vocab {row['vocab_size']:>6,}: {row['tokens']:>5,} tokens, "
            f"{row['chars_per_token']:>5.2f} chars/tok, "
            f"{row['total_params'] / 1e6:>5.1f}M params, "
            f"cost {row['relative_cost'] / 1e9:>8.1f}  ({seconds:.0f}s)",
            flush=True,
        )
        if size == args.vocab_size:
            kept = tokenizer

    # Qwen as the reference point the plan's claim was made against.
    baseline = None
    try:
        from transformers import AutoTokenizer

        qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        baseline = measure(lambda t: qwen.encode(t), qwen.vocab_size)
        print(
            f"  Qwen  {baseline['vocab_size']:>6,}: {baseline['tokens']:>5,} tokens, "
            f"{baseline['chars_per_token']:>5.2f} chars/tok, "
            f"{baseline['total_params'] / 1e6:>5.1f}M params, "
            f"cost {baseline['relative_cost'] / 1e9:>8.1f}"
        )
    except Exception as exc:  # noqa: BLE001 — the comparison is informative, not required
        print(f"  [WARN] Qwen tokenizer unavailable for comparison: {exc}")

    best = min(rows, key=lambda r: r["relative_cost"])
    print(f"\n  Lowest training cost: vocab {best['vocab_size']:,}")
    if baseline:
        saving = (1 - best["relative_cost"] / baseline["relative_cost"]) * 100
        print(f"  {saving:.1f}% cheaper than Qwen's vocab at the same d_model\n")

    if kept is None:
        raise SystemExit(f"FATAL: --vocab-size {args.vocab_size} was not among {sizes}")
    save(kept, args.out)
    print(f"  Saved vocab-{args.vocab_size:,} tokenizer to {args.out}/")

    report = {
        "documents_sampled": documents,
        "sweep": rows,
        "qwen": baseline,
        "kept": args.vocab_size,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  Wrote {args.report}")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write("## Phase 2 — tokenizer vocab sweep\n\n")
            fh.write(f"Trained on {documents:,} sampled documents.\n\n")
            fh.write("| Vocab | Tokens on probes | chars/token | Model params | Relative cost |\n")
            fh.write("| ---: | ---: | ---: | ---: | ---: |\n")
            for row in rows + ([baseline] if baseline else []):
                label = f"{row['vocab_size']:,}"
                if baseline and row is baseline:
                    label += " (Qwen)"
                fh.write(
                    f"| {label} | {row['tokens']:,} | {row['chars_per_token']} | "
                    f"{row['total_params'] / 1e6:.1f}M | {row['relative_cost'] / 1e9:.1f} |\n"
                )
            fh.write(
                f"\nLowest cost: **vocab {best['vocab_size']:,}**. Kept: {args.vocab_size:,}.\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
