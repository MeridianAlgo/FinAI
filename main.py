#!/usr/bin/env python3
import sys
import argparse
from src.core.finai import FinAI
from src.config import Config


def main():
    parser = argparse.ArgumentParser(description="FinAI (Local LLM) CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train from local text file
    p_train = subparsers.add_parser("train", help="Train from a local .txt file (plain text corpus)")
    p_train.add_argument("dataset_file", help="Path to .txt dataset file")
    p_train.add_argument("--steps", type=int, default=None, help="Training steps (tokens batches)")
    p_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p_train.add_argument("--lr", type=float, default=None, help="Learning rate")
    p_train.add_argument("--block-size", type=int, default=None, help="Context window (tokens)")
    p_train.add_argument("--cpu", action="store_true", help="Force CPU training")
    p_train.add_argument("--accelerate", choices=["auto", "on", "off"], default="auto", help="Use Hugging Face Accelerate (auto/on/off)")
    p_train.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p_train.add_argument("--mixed-precision", choices=["auto", "no", "fp16", "bf16"], default="auto", help="Mixed precision (auto/no/fp16/bf16)")
    p_train.add_argument("--one-epoch", action="store_true", help="Train for one full epoch (automatically calculates steps based on dataset size)")

    # Chat
    subparsers.add_parser("chat", help="Interactive chat with the trained model")

    # Generate from prompt
    p_gen = subparsers.add_parser("generate", help="Generate from a prompt")
    p_gen.add_argument("prompt", help="Prompt text")

    # Train directly from a Hugging Face dataset id (exports to txt under the hood)
    p_train_hf = subparsers.add_parser(
        "train_hf",
        help="Download an HF dataset, export to local .txt, and train end-to-end",
    )
    p_train_hf.add_argument("dataset_id", help="HF dataset id, e.g. PatronusAI/financebench")
    p_train_hf.add_argument("--split", default="train", help="Split name (default: train)")
    p_train_hf.add_argument("--max", type=int, default=None, help="Max items to export (optional)")
    p_train_hf.add_argument("--steps", type=int, default=None, help="Training steps (optional)")
    p_train_hf.add_argument("--batch-size", type=int, default=None, help="Batch size (optional)")
    p_train_hf.add_argument("--lr", type=float, default=None, help="Learning rate (optional)")
    p_train_hf.add_argument("--block-size", type=int, default=None, help="Context window tokens (optional)")
    p_train_hf.add_argument("--cpu", action="store_true", help="Force CPU training")
    p_train_hf.add_argument("--accelerate", choices=["auto", "on", "off"], default="auto", help="Use Hugging Face Accelerate (auto/on/off)")
    p_train_hf.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p_train_hf.add_argument("--mixed-precision", choices=["auto", "no", "fp16", "bf16"], default="auto", help="Mixed precision (auto/no/fp16/bf16)")
    p_train_hf.add_argument("--one-epoch", action="store_true", help="Train for one full epoch (automatically calculates steps based on dataset size)")

    args = parser.parse_args()
    finai = FinAI()

    if args.command == "train":
        finai.train_from_file(
            filepath=args.dataset_file,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            block_size=args.block_size,
            use_gpu=(not args.cpu),
            use_accelerate=(True if args.accelerate == "on" else False if args.accelerate == "off" else "auto"),
            grad_accum_steps=args.grad_accum,
            mixed_precision=args.mixed_precision,
            one_epoch=args.one_epoch,
        )
        return 0

    if args.command == "generate":
        if finai.initialize():
            out = finai.generate_response(args.prompt)
            print(out)
        return 0

    if args.command == "chat":
        finai.run()
        return 0

    if args.command == "train_hf":
        # Lazy import datasets for optional HF workflow
        try:
            from datasets import load_dataset  # type: ignore
        except Exception:
            print("Missing dependency: pip install datasets")
            return 1

        def _sanitize(name: str) -> str:
            return name.replace('/', '_').replace('\\', '_').replace('-', '_')

        print(f"Downloading dataset: {args.dataset_id}")
        ds = load_dataset(args.dataset_id)
        # Defaults from Config when flags are omitted
        cfg = Config()
        split_arg = args.split or cfg.HF_DEFAULT_SPLIT
        split = split_arg if split_arg in ds else list(ds.keys())[0]
        max_items = args.max if args.max is not None else cfg.EXPORT_MAX
        steps = args.steps if args.steps is not None else cfg.TRAIN_STEPS
        batch_size = args.batch_size if args.batch_size is not None else cfg.BATCH_SIZE
        lr = args.lr if args.lr is not None else cfg.LEARNING_RATE
        block_size = args.block_size if args.block_size is not None else cfg.BLOCK_SIZE
        data = ds[split]

        # Extract texts
        common_fields = [
            'text', 'tweet', 'content', 'input', 'question', 'instruction', 'prompt', 'title', 'selftext', 'answer', 'response'
        ]
        texts = []
        for item in data:
            txt = None
            for f in common_fields:
                if f in item and isinstance(item[f], str) and item[f].strip():
                    txt = item[f].strip()
                    break
            if not txt:
                parts = [str(v).strip() for k, v in item.items() if isinstance(v, str) and v.strip()]
                if parts:
                    txt = " ".join(parts)
            if txt and len(txt) >= 5:
                texts.append(txt)
            if max_items and len(texts) >= max_items:
                break

        if not texts:
            print("No texts extracted from dataset; aborting.")
            return 1

        # Write to local txt
        import os
        os.makedirs('datasets', exist_ok=True)
        out_file = os.path.join('datasets', f"hf_{_sanitize(args.dataset_id)}.txt")
        with open(out_file, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(t.replace('\r', ' ').strip() + "\n\n")
        print(f"Exported {len(texts)} texts to {out_file}")

        # Train on local txt
        finai.train_from_file(
            filepath=out_file,
            steps=steps,
            batch_size=batch_size,
            learning_rate=lr,
            block_size=block_size,
            use_gpu=(not args.cpu),
            use_accelerate=(True if args.accelerate == "on" else False if args.accelerate == "off" else "auto"),
            grad_accum_steps=args.grad_accum,
            mixed_precision=args.mixed_precision,
            one_epoch=args.one_epoch,
        )
        # Update CSV tracking: move dataset to trained_datasets.csv; fallback row if not in datasets.csv
        import csv, os
        DATASETS_CSV = "datasets.csv"
        TRAINED_CSV = "trained_datasets.csv"
        headers = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']
        os.makedirs(os.path.dirname(TRAINED_CSV) or '.', exist_ok=True)

        def _load_csv(path):
            if not os.path.exists(path):
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                return []
            with open(path, 'r', newline='', encoding='utf-8') as f:
                return list(csv.DictReader(f))

        def _save_csv(path, rows):
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                w.writerows(rows)

        datasets_rows = _load_csv(DATASETS_CSV)
        trained_rows = _load_csv(TRAINED_CSV)
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_path = 'models/finai_gpt.pt'

        found = False
        remaining = []
        for row in datasets_rows:
            if row.get('name') == args.dataset_id and not found:
                found = True
                trained_rows.append({
                    'name': row.get('name', ''),
                    'config': row.get('config', ''),
                    'split': row.get('split', 'train'),
                    'date_trained': now,
                    'model_path': model_path,
                    'status': 'completed',
                })
            else:
                remaining.append(row)

        if not found:
            # Fallback: add FinanceInc/auditor_sentiment row as requested
            trained_rows.append({
                'name': 'FinanceInc/auditor_sentiment',
                'config': '',
                'split': 'train',
                'date_trained': now,
                'model_path': model_path,
                'status': 'completed',
            })

        _save_csv(DATASETS_CSV, remaining)
        _save_csv(TRAINED_CSV, trained_rows)

        return 0

    # Default to chat
    finai.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
