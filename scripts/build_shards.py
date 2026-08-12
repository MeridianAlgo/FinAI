"""Phase 2: tokenize the corpus into uint16 shards for memmap training.

Replaces live streaming at train time. Per the comments in ``train.yml``, streaming ~25
datasets concurrently with per-stream shuffle buffers is what drove RSS past 15 GB, forced
``SHUFFLE_BUFFER`` down to 128, and made fineweb-edu unusable. Tokenizing once up front
gives constant memory, near-zero data cost during training, and a reproducible ordering.

Two passes, because mixing and memory safety pull in opposite directions:

* **Pass A** visits one dataset at a time and writes its tokens to a staging ``.bin``. Peak
  memory is a single Arrow row-group, not twenty-five.
* **Pass B** interleaves the staging files into shards by sampling a source per block
  according to the mix weights. Mixing over tokenized files on disk costs nothing, whereas
  mixing at stream time is exactly what cost 15 GB.

Held-out validation is carved out per domain, so Phase 3 can watch finance and general
perplexity separately -- a model improving on finance while rotting on general English is a
specific failure worth being able to see.

Usage:
    python scripts/build_shards.py --target-tokens 300_000_000
    python scripts/build_shards.py --target-tokens 300_000_000 --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone

import numpy as np

from meridian.data.corpus import dataset_specs, iter_documents, slug

EOS_TOKEN = "<|endoftext|>"
DTYPE = np.uint16  # vocab is 16,384, so 2 bytes/token halves the corpus on disk and in RAM


def tokenize_dataset(spec, tokenizer, eos_id, target_tokens, staging_dir, batch_size):
    """Pass A: stream one dataset and append its tokens to a staging file."""
    path = os.path.join(staging_dir, f"{slug(spec['name'])}.bin")
    written = 0
    started = time.perf_counter()

    # Documents are batch-encoded: the Rust tokenizer parallelizes across a batch, and
    # per-call overhead dominates otherwise.
    with open(path, "wb") as fh:
        batch: list[str] = []

        def flush() -> int:
            if not batch:
                return 0
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            flat: list[int] = []
            for ids in encoded:
                flat.extend(ids)
                flat.append(eos_id)  # documents are independent; EOS marks the boundary
            arr = np.array(flat, dtype=DTYPE)
            arr.tofile(fh)
            batch.clear()
            return len(arr)

        # Character budget is a guard against pulling far more text than we can use: at
        # roughly 4 chars/token, 6x the token target is ample headroom.
        for text in iter_documents(
            spec, EOS_TOKEN, max_documents=10**9, max_chars=target_tokens * 6
        ):
            batch.append(text)
            if len(batch) >= batch_size:
                written += flush()
                if written >= target_tokens:
                    break
        written += flush()

    elapsed = time.perf_counter() - started
    rate = written / elapsed if elapsed > 0 else 0
    print(
        f"    {spec['name']:<48} {written:>12,} tok  ({elapsed:>5.0f}s, {rate:>8,.0f} tok/s)",
        flush=True,
    )
    return {"path": path, "tokens": written, "domain": spec["domain"], "weight": spec["weight"]}


def write_split(sources, out_path, total_tokens, block, rng, shard_tokens=None):
    """Pass B: interleave staging files into one output, sampling sources by weight.

    Copying in blocks rather than single tokens keeps each document's tokens contiguous,
    which is what the model needs to see; the interleaving happens between blocks.
    """
    live = [s for s in sources if s["remaining"] > 0]
    if not live:
        return []

    shards, written_total = [], 0
    shard_index, shard_written, handle = 0, 0, None

    def open_shard():
        nonlocal handle, shard_written
        name = out_path if shard_tokens is None else f"{out_path}_{shard_index:05d}.bin"
        handle = open(name, "wb")
        shard_written = 0
        return name

    current = open_shard()
    shards.append(current)

    while written_total < total_tokens and live:
        weights = [s["weight"] for s in live]
        source = rng.choices(live, weights=weights, k=1)[0]

        take = min(block, source["remaining"], total_tokens - written_total)
        chunk = np.fromfile(source["path"], dtype=DTYPE, count=take, offset=source["offset"] * 2)
        if chunk.size == 0:
            source["remaining"] = 0
            live = [s for s in sources if s["remaining"] > 0]
            continue

        chunk.tofile(handle)
        source["offset"] += chunk.size
        source["remaining"] -= chunk.size
        written_total += chunk.size
        shard_written += chunk.size

        if source["remaining"] <= 0:
            live = [s for s in sources if s["remaining"] > 0]

        if shard_tokens is not None and shard_written >= shard_tokens:
            handle.close()
            shard_index += 1
            current = open_shard()
            shards.append(current)

    handle.close()
    if shard_tokens is not None and shard_written == 0:
        os.remove(shards.pop())
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--target-tokens", type=int, default=300_000_000)
    parser.add_argument("--val-tokens-per-domain", type=int, default=2_000_000)
    parser.add_argument("--shard-tokens", type=int, default=50_000_000)
    parser.add_argument("--interleave-block", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", default="corpus")
    parser.add_argument("--staging", default="corpus_staging")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include-heavy", action="store_true", default=True)
    parser.add_argument("--no-include-heavy", dest="include_heavy", action="store_false")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default="meridianal/FinAI-corpus")
    # Private by default: these shards are a derivative of the source datasets and are
    # reconstructible back to their text given the tokenizer, so republishing them publicly
    # is a licensing decision to make deliberately rather than a side effect of a build.
    parser.add_argument("--public", action="store_true", help="Publish the dataset repo publicly")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    if tokenizer.vocab_size > np.iinfo(DTYPE).max:
        raise SystemExit(f"vocab {tokenizer.vocab_size} does not fit in {DTYPE.__name__}")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.staging, exist_ok=True)
    specs = dataset_specs(include_heavy=args.include_heavy)

    # Over-collect: pass B samples by weight and stops at the target, so any single source
    # running dry early would skew the realized mix.
    val_total = args.val_tokens_per_domain * 2
    grand_total = args.target_tokens + val_total
    headroom = 1.35

    print("=" * 78)
    print("  Phase 2 — tokenizing corpus to uint16 shards")
    print("=" * 78)
    print(f"  Target: {args.target_tokens:,} train + {val_total:,} val tokens\n")
    print("  Pass A — per-dataset tokenization")

    staged = []
    for spec in specs:
        quota = int(grand_total * spec["weight"] * headroom) + 100_000
        staged.append(
            tokenize_dataset(spec, tokenizer, eos_id, quota, args.staging, args.batch_size)
        )

    staged = [s for s in staged if s["tokens"] > 0]
    if not staged:
        raise SystemExit("FATAL: no dataset produced tokens")
    print(f"\n  Staged {sum(s['tokens'] for s in staged):,} tokens from {len(staged)} datasets\n")

    rng = random.Random(args.seed)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": args.tokenizer,
        "vocab_size": int(tokenizer.vocab_size),
        "dtype": DTYPE.__name__,
        "eos_token_id": int(eos_id),
        "seed": args.seed,
        "splits": {},
        "sources": [
            {"name": s["path"], "tokens": s["tokens"], "domain": s["domain"]} for s in staged
        ],
    }

    print("  Pass B — validation splits (held out first, never trained on)")
    for domain in ("finance", "general"):
        sources = [
            {**s, "offset": 0, "remaining": s["tokens"]} for s in staged if s["domain"] == domain
        ]
        if not sources:
            continue
        path = os.path.join(args.out, f"val_{domain}.bin")
        write_split(sources, path, args.val_tokens_per_domain, args.interleave_block, rng)
        tokens = os.path.getsize(path) // 2
        manifest["splits"][f"val_{domain}"] = {"files": [os.path.basename(path)], "tokens": tokens}
        print(f"    val_{domain}: {tokens:,} tokens", flush=True)
        # Advance the staging offsets so training never sees these tokens.
        for src in sources:
            for original in staged:
                if original["path"] == src["path"]:
                    original["val_offset"] = src["offset"]

    print("\n  Pass B — training shards")
    train_sources = [
        {
            **s,
            "offset": s.get("val_offset", 0),
            "remaining": s["tokens"] - s.get("val_offset", 0),
            "start": s.get("val_offset", 0),
        }
        for s in staged
    ]
    shards = write_split(
        train_sources,
        os.path.join(args.out, "train"),
        args.target_tokens,
        args.interleave_block,
        rng,
        shard_tokens=args.shard_tokens,
    )

    # A dataset that runs dry has its share silently redistributed to whatever is still
    # live, so the realized mix can drift well away from the configured weights. Record
    # both per source and flag the exhausted ones rather than letting it pass unseen.
    consumed_total = sum(s["offset"] - s["start"] for s in train_sources) or 1
    realized_sources, exhausted = [], []
    for source in train_sources:
        consumed = source["offset"] - source["start"]
        entry = {
            "name": source["path"],
            "domain": source["domain"],
            "target_weight": round(source["weight"], 5),
            "realized_weight": round(consumed / consumed_total, 5),
            "tokens": consumed,
            "exhausted": source["remaining"] <= 0,
        }
        realized_sources.append(entry)
        if entry["exhausted"]:
            exhausted.append(entry)
    manifest["realized_sources"] = realized_sources

    if exhausted:
        print(f"\n    [WARN] {len(exhausted)} dataset(s) exhausted; mix skewed toward the rest:")
        for entry in sorted(exhausted, key=lambda e: -e["tokens"])[:8]:
            print(
                f"      {os.path.basename(entry['name']):<44} "
                f"target {entry['target_weight']:.4f} -> realized {entry['realized_weight']:.4f}"
            )
        print()
    train_tokens = sum(os.path.getsize(p) // 2 for p in shards)
    manifest["splits"]["train"] = {
        "files": [os.path.basename(p) for p in shards],
        "tokens": train_tokens,
    }
    for path in shards:
        print(f"    {os.path.basename(path)}: {os.path.getsize(path) // 2:,} tokens", flush=True)

    # Domain mix of what the model will actually read, derived from realized consumption
    # rather than configured weights.
    by_domain: dict[str, int] = {}
    for entry in realized_sources:
        by_domain[entry["domain"]] = by_domain.get(entry["domain"], 0) + entry["tokens"]
    manifest["domain_mix"] = {k: round(v / consumed_total, 4) for k, v in by_domain.items()}

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n  Train: {train_tokens:,} tokens across {len(shards)} shards")
    print(f"  Domain mix: {manifest['domain_mix']}")
    print(f"  Wrote {args.out}/manifest.json")

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write("## Phase 2 — corpus shards\n\n")
            fh.write(f"- **{train_tokens:,}** training tokens in {len(shards)} shards\n")
            for name, split in manifest["splits"].items():
                if name.startswith("val"):
                    fh.write(f"- **{split['tokens']:,}** held-out {name} tokens\n")
            fh.write(f"- Domain mix: `{manifest['domain_mix']}`\n")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=not args.public)
        api.upload_folder(
            folder_path=args.out,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Corpus: {train_tokens:,} train tokens [skip ci]",
        )
        print(f"  Pushed to https://huggingface.co/datasets/{args.repo_id}")

    if not args.keep_staging:
        shutil.rmtree(args.staging, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
