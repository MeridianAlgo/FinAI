#!/usr/bin/env python3
"""
Export a Hugging Face dataset to a local .txt corpus for FinAI training.
Usage:
  python scripts/export_hf_to_txt.py <dataset_id> [--split SPLIT] [--max N]
Example:
  python scripts/export_hf_to_txt.py zeroshot/twitter-financial-news-sentiment --split train --max 200000
"""
import argparse
import os
from typing import List

try:
    from datasets import load_dataset  # pip install datasets
except Exception as e:
    raise SystemExit("Missing dependency: pip install datasets")

COMMON_TEXT_FIELDS: List[str] = [
    'text', 'tweet', 'content', 'input', 'question', 'instruction', 'prompt', 'title', 'selftext'
]

def extract_texts(ds, split: str, max_items: int = None) -> List[str]:
    if split not in ds:
        # pick first available split
        split = list(ds.keys())[0]
    data = ds[split]
    out: List[str] = []
    for item in data:
        text = None
        # try common fields first
        for field in COMMON_TEXT_FIELDS:
            if field in item and isinstance(item[field], str) and item[field].strip():
                text = item[field].strip()
                break
        # fallback: combine all string-like fields
        if not text:
            parts = []
            for k, v in item.items():
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            if parts:
                text = " ".join(parts)
        if text and len(text) >= 5:
            out.append(text)
        if max_items and len(out) >= max_items:
            break
    return out


def sanitize_filename(s: str) -> str:
    return s.replace('/', '_').replace('\\', '_').replace('-', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_id', help='Hugging Face dataset id, e.g. zeroshot/twitter-financial-news-sentiment')
    parser.add_argument('--split', default='train', help='split name (default: train)')
    parser.add_argument('--max', type=int, default=None, help='max number of items to export')
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id)

    print(f"Extracting texts from split: {args.split}")
    texts = extract_texts(ds, args.split, max_items=args.max)
    print(f"Collected {len(texts)} texts")

    os.makedirs('datasets', exist_ok=True)
    out_file = os.path.join('datasets', f"hf_{sanitize_filename(args.dataset_id)}.txt")

    print(f"Writing to {out_file} ...")
    with open(out_file, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t.replace('\r', ' ').strip() + "\n\n")

    print(f"Done. Saved: {out_file}")


if __name__ == '__main__':
    main()
