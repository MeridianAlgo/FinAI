"""Check every dataset spec against the live Hub schema.

The first Phase 2 corpus build exposed that 14 of the 27 configured datasets produced zero
documents: their spec named a column the dataset does not have, so ``_format_text``
returned "" and every row was silently dropped. Nothing failed loudly — the mix just
quietly became half of what it claimed to be, in the trainer as well as the corpus builder.

This makes that class of failure loud. It queries the HF datasets-server (metadata only, no
downloads) and reports, per spec, whether the config/split exists and whether the configured
text field is actually a column.

Exits non-zero if any spec is broken, so CI can gate on it.

Usage:
    python scripts/validate_datasets.py
    python scripts/validate_datasets.py --json docs/benchmarks/datasets.json
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request

from meridian.data.corpus import dataset_specs

SERVER = "https://datasets-server.huggingface.co"


def api(path: str, **params) -> dict:
    url = f"{SERVER}/{path}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except Exception as exc:  # noqa: BLE001 — a probe failure is a result, not a crash
        return {"error": str(exc)[:120]}


def check(spec: dict) -> dict:
    name = spec["name"]
    want_config = spec.get("config") or "default"
    want_split = spec["split"]
    text_field = spec.get("text_field")

    result = {
        "name": name,
        "config": want_config,
        "split": want_split,
        "text_field": text_field,
        "ok": False,
        "problem": None,
        "columns": [],
    }

    # Sources with a custom loader (FinDB reads a SQLite file out of a git repo) are not on
    # the Hub, so there is no schema here to check.
    if spec.get("loader"):
        result["ok"] = True
        result["problem"] = f"skipped: custom loader '{spec['loader']}'"
        return result

    splits = api("splits", dataset=name)
    if "splits" not in splits:
        result["problem"] = f"splits unavailable: {splits.get('error', 'unknown')}"
        return result

    pairs = {(entry["config"], entry["split"]) for entry in splits["splits"]}
    if (want_config, want_split) not in pairs:
        available = sorted({f"{c}/{s}" for c, s in pairs})[:6]
        result["problem"] = f"no such config/split; available: {', '.join(available)}"
        return result

    rows = api("first-rows", dataset=name, config=want_config, split=want_split)
    columns = [feature["name"] for feature in rows.get("features", [])]
    result["columns"] = columns
    if not columns:
        result["problem"] = f"no readable columns: {rows.get('error', 'empty')}"
        return result

    missing = [
        field
        for field in (text_field, spec.get("instruction_field"), spec.get("label_field"))
        if field and field not in columns
    ]
    if missing:
        result["problem"] = f"missing column(s) {missing}; has {columns[:6]}"
        return result

    result["ok"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--include-heavy", action="store_true", default=True)
    args = parser.parse_args()

    specs = dataset_specs(include_heavy=args.include_heavy)
    print(f"Validating {len(specs)} dataset specs against the Hub\n")

    results = []
    for spec in specs:
        outcome = check(spec)
        results.append(outcome)
        mark = "ok  " if outcome["ok"] else "FAIL"
        line = f"  [{mark}] {outcome['name']:<44} {outcome['config']}/{outcome['split']}"
        if not outcome["ok"]:
            line += f"\n           {outcome['problem']}"
        print(line, flush=True)

    broken = [r for r in results if not r["ok"]]
    weight_ok = sum(s["weight"] for s, r in zip(specs, results) if r["ok"])
    print(
        f"\n  {len(results) - len(broken)}/{len(results)} specs usable, "
        f"covering {weight_ok * 100:.1f}% of the configured mix weight"
    )

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Wrote {args.json_out}")

    if broken:
        print(f"\n  {len(broken)} broken spec(s) — these contribute nothing to training.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
