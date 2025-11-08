#!/usr/bin/env python3
"""
Sort datasets.csv by approximate dataset size (smallest to largest).

Strategy:
- Try to fetch dataset_infos.json from Hugging Face Hub for each dataset.
- Use per-config info if provided; otherwise pick the first available config.
- Prefer 'dataset_size' (bytes). If unavailable, sum split sizes if present.
- If sizes are unavailable, fall back to 'num_examples' for the split.
- If nothing available, place item at the end, preserving relative order.

Requires: datasets, huggingface_hub (installed with datasets)
"""
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets.csv"

HEADERS = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']


def read_rows(path: Path):
    with path.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def get_size_for_dataset(repo_id: str, config: Optional[str], split: str) -> Tuple[int, int]:
    """Return a tuple (size_bytes, num_examples) for sorting.
    If not available, returns (inf, inf) so they go to the end.
    """
    try:
        # Download dataset_infos.json from the repo
        infos_path = hf_hub_download(repo_id=repo_id, filename='dataset_infos.json', repo_type='dataset')
        with open(infos_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return (float('inf'), float('inf'))

    # Select config
    if not isinstance(data, dict) or not data:
        return (float('inf'), float('inf'))

    cfg_key = None
    if config and config in data:
        cfg_key = config
    else:
        # Pick first config in stable order
        cfg_key = sorted(list(data.keys()))[0]

    info = data.get(cfg_key, {})

    # Prefer dataset_size in bytes
    size_bytes = info.get('dataset_size')

    # If not present, try summing split sizes
    if size_bytes is None:
        splits = info.get('splits') or {}
        total = 0
        found = False
        for s_name, s_info in splits.items():
            sz = s_info.get('num_bytes') or s_info.get('size_in_bytes')
            if isinstance(sz, int):
                total += sz
                found = True
        if found:
            size_bytes = total

    # If still missing, fallback to num_examples
    num_examples = None
    splits = info.get('splits') or {}
    if split in splits and isinstance(splits[split], dict):
        num_examples = splits[split].get('num_examples')

    # Defaulting when missing
    if size_bytes is None and num_examples is None:
        return (float('inf'), float('inf'))

    if size_bytes is None:
        size_bytes = float('inf')  # will sort after items with known byte size

    if num_examples is None:
        num_examples = float('inf')

    return (size_bytes, num_examples)


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found")
        sys.exit(1)

    rows = read_rows(CSV_PATH)

    # Deduplicate by name+config+split preserving first occurrence
    seen = set()
    deduped = []
    for r in rows:
        key = (r.get('name', ''), r.get('config', ''), r.get('split', 'train'))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Compute size info
    sized = []
    for r in deduped:
        repo_id = (r.get('name') or '').strip()
        config = (r.get('config') or '').strip() or None
        split = (r.get('split') or 'train').strip() or 'train'
        size_bytes, num_examples = get_size_for_dataset(repo_id, config, split)
        sized.append((size_bytes, num_examples, r))

    # Sort: by size_bytes first, then num_examples, with inf at the end
    sized.sort(key=lambda x: (x[0], x[1]))

    sorted_rows = [r for _, _, r in sized]

    write_rows(CSV_PATH, sorted_rows)
    print(f"✓ datasets.csv sorted by size (smallest to largest)")


if __name__ == '__main__':
    main()
