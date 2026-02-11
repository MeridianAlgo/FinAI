"""Finance-Focused Data Pipeline.

Curriculum-aware dataset mixing with finance, math, and general knowledge.
Uses streaming to avoid downloading massive datasets.

Dataset mix:
 - 40% FinanceAlpaca (financial QA & instructions)  
 - 30% OpenMathInstruct (math reasoning — critical for finance)
 - 30% FineWeb-Edu (general knowledge foundation)

This mix ensures the model excels at finance + math while maintaining
broad language understanding.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import torch
from datasets import load_dataset


class FinanceDataPipeline:
    """Streaming data pipeline with finance-focused curriculum mixing.

    Yields tokenized examples from multiple datasets with configurable
    mixing ratios. Supports skip-ahead for continual training resume.
    """

    # Dataset configurations: (name, config, text_field, weight)
    DATASETS = [
        {
            "name": "gbharti/finance-alpaca",
            "config": None,
            "split": "train",
            "text_field": "output",  # Financial instructions
            "instruction_field": "instruction",
            "weight": 0.40,
        },
        {
            "name": "nvidia/OpenMathInstruct-2",
            "config": None,
            "split": "train_1M",
            "text_field": "generated_solution",
            "instruction_field": "problem",
            "weight": 0.30,
        },
        {
            "name": "HuggingFaceFW/fineweb-edu",
            "config": "default",
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.30,
        },
    ]

    def __init__(
        self,
        tokenizer,
        block_size: int = 512,
        skip_items: int = 0,
        max_bytes_per_run: int = 50 * 1024 * 1024,  # 50MB per hourly run
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.skip_items = skip_items
        self.max_bytes_per_run = max_bytes_per_run
        self.items_processed = 0

    def _load_stream(self, ds_config: dict):
        """Load a streaming dataset with retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "path": ds_config["name"],
                    "split": ds_config["split"],
                    "streaming": True,
                }
                if ds_config["config"]:
                    kwargs["name"] = ds_config["config"]

                dataset = load_dataset(**kwargs)
                return dataset
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Retry {attempt + 1} for {ds_config['name']}: {e}")
                    time.sleep(5)
                else:
                    print(f"  ✗ Failed to load {ds_config['name']}: {e}")
                    return None

    def _format_text(self, item: dict, ds_config: dict) -> str:
        """Format a dataset item into training text.

        For instruction datasets, creates instruction-response format.
        For plain text datasets, uses raw text.
        """
        instruction = ""
        if ds_config["instruction_field"] and ds_config["instruction_field"] in item:
            instruction = item[ds_config["instruction_field"]]

        text = item.get(ds_config["text_field"], "")
        if not isinstance(text, str):
            text = str(text) if text else ""

        if instruction and text:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{text}"
        elif text:
            return text
        return ""

    def stream(self) -> Iterator[dict]:
        """Yield tokenized examples from mixed datasets."""
        # Load all dataset streams
        streams = []
        for ds_config in self.DATASETS:
            dataset = self._load_stream(ds_config)
            if dataset is not None:
                if self.skip_items > 0:
                    per_ds_skip = int(self.skip_items * ds_config["weight"])
                    dataset = dataset.skip(per_ds_skip)
                streams.append((iter(dataset), ds_config))
                print(f"  ✓ Loaded {ds_config['name']} (weight: {ds_config['weight']})")

        if not streams:
            print("  ✗ No datasets loaded! Falling back to synthetic data.")
            yield from self._synthetic_fallback()
            return

        total_bytes = 0
        # Round-robin with weights
        stream_indices = list(range(len(streams)))
        weights = [self.DATASETS[i]["weight"] for i in range(len(streams))]

        # Create weighted order: repeat indices based on weight
        weighted_order = []
        for idx, w in zip(stream_indices, weights):
            count = max(1, int(w * 10))
            weighted_order.extend([idx] * count)

        order_idx = 0
        exhausted = set()

        while len(exhausted) < len(streams):
            # Pick next stream based on weighted order
            stream_idx = weighted_order[order_idx % len(weighted_order)]
            order_idx += 1

            if stream_idx in exhausted:
                continue

            stream_iter, ds_config = streams[stream_idx]

            try:
                item = next(stream_iter)
            except StopIteration:
                exhausted.add(stream_idx)
                continue

            text = self._format_text(item, ds_config)
            if not text or not text.strip():
                continue

            text_bytes = len(text.encode("utf-8"))
            if total_bytes + text_bytes > self.max_bytes_per_run:
                print(f"  [INFO] Byte limit reached ({total_bytes / 1e6:.1f}MB). Ending.")
                return

            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.block_size,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Mask padding tokens in labels
            if self.tokenizer.pad_token_id is not None:
                labels[input_ids == self.tokenizer.pad_token_id] = -100

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "processed_idx": self.skip_items + self.items_processed,
            }

            total_bytes += text_bytes
            self.items_processed += 1

    def _synthetic_fallback(self) -> Iterator[dict]:
        """Generate synthetic data if all datasets fail to load."""
        vocab_size = self.tokenizer.vocab_size
        for i in range(100):
            input_ids = torch.randint(0, vocab_size, (self.block_size,))
            yield {
                "input_ids": input_ids,
                "attention_mask": torch.ones(self.block_size, dtype=torch.long),
                "labels": input_ids.clone(),
                "processed_idx": i,
            }
            self.items_processed += 1


class _IterableDataset(torch.utils.data.IterableDataset):
    """Wraps a generator into a PyTorch IterableDataset."""

    def __init__(self, gen_fn):
        self.gen_fn = gen_fn

    def __iter__(self):
        return self.gen_fn()


def create_dataloader(
    tokenizer,
    batch_size: int = 2,
    block_size: int = 512,
    skip_items: int = 0,
    max_bytes: int = 50 * 1024 * 1024,
) -> torch.utils.data.DataLoader:
    """Create the finance-focused training DataLoader."""
    pipeline = FinanceDataPipeline(
        tokenizer=tokenizer,
        block_size=block_size,
        skip_items=skip_items,
        max_bytes_per_run=max_bytes,
    )

    dataset = _IterableDataset(pipeline.stream)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def create_smoke_dataloader(
    vocab_size: int, batch_size: int, block_size: int
) -> torch.utils.data.DataLoader:
    """Create a synthetic dataloader for CI smoke tests."""

    def gen():
        for _ in range(500):
            input_ids = torch.randint(0, vocab_size, (block_size,), dtype=torch.long)
            yield {
                "input_ids": input_ids,
                "attention_mask": torch.ones(block_size, dtype=torch.long),
                "labels": input_ids.clone(),
            }

    dataset = _IterableDataset(gen)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
