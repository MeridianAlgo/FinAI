"""Finance-Focused Data Pipeline.

Shuffle-seed-based dataset mixing with finance, math, and general knowledge.
Uses streaming to avoid downloading massive datasets. Each training run derives
a shuffle seed from the processed_items counter so it samples a different region
of each dataset without sequential .skip() overhead (which downloads thousands
of items before yielding a single training example).

Dataset mix (v6.0.0 weights):
 - 26% gbharti/finance-alpaca        (financial Q&A instructions)
 - 18% sujet-ai/Sujet-Finance-Instruct-177k  (high-quality finance instruct)
 - 15% nvidia/OpenMathInstruct-2     (math reasoning for quantitative finance)
 - 12% HuggingFaceFW/fineweb-edu    (general knowledge foundation)
 - 05% yahma/alpaca-cleaned               (general instruction format)
 - ~24% FinanceMTEB + FinGPT + misc  (sentiment, ESG, fraud, FLS, events, etc.)

Weights rebalanced in v6.0.0:
 - Reduced OpenMathInstruct 0.25→0.15 (math training caused factual confusion)
 - Increased Sujet-Finance-Instruct 0.12→0.18 (highest quality finance instruct)
 - Added yahma/alpaca-cleaned 0.05 (improves response format consistency)
"""

from __future__ import annotations

import os
import random
import time
from typing import Iterator

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
            "weight": 0.26,
        },
        {
            "name": "nvidia/OpenMathInstruct-2",
            "config": None,
            "split": "train_1M",
            "text_field": "generated_solution",
            "instruction_field": "problem",
            "weight": 0.15,
            # "heavy": multi-GB Parquet row-groups. Only a capped number of heavy
            # datasets stream concurrently (see MAX_HEAVY_CONCURRENT) to bound RAM.
            "heavy": True,
        },
        {
            "name": "HuggingFaceFW/fineweb-edu",
            "config": "default",
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.12,
            "heavy": True,
        },
        {
            "name": "FinanceMTEB/financial_phrasebank",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following financial sentence.\n\nSentence:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/FinSent",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following sentence.\n\nSentence:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/OpenFinDataSentiment",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following financial text.\n\nText:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/FiQA_ABSA",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) expressed in the following finance-related text.\n\nText:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/SemEva2017_Headline",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following headline.\n\nHeadline:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/ESG",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "prompt_template": "Given the following text, classify it into the appropriate ESG-related category.\n\nText:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/FOMC",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.008,
        },
        {
            "name": "FinanceMTEB/FinancialFraud",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "prompt_template": "Determine whether the following case description indicates potential financial fraud. Answer with a short label.\n\nText:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/Complaints",
            "config": None,
            "split": "test",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "prompt_template": "Classify the product/category for the following financial complaint.\n\nComplaint:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/FLS",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.008,
        },
        {
            "name": "FinanceMTEB/FinFE",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.008,
        },
        {
            "name": "FinanceMTEB/FinEvaSentiment",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following text.\n\nText:\n{text}",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/FinChinaSentiment",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following text.\n\nText:\n{text}",
            "weight": 0.008,
        },
        {
            "name": "FinanceMTEB/TradeTheEventNews",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.008,
        },
        {
            "name": "FinanceMTEB/TradeTheEventEncyclopedia",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.008,
        },
        {
            "name": "FinanceMTEB/AlphaFin",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.008,
        },
        {
            "name": "FinanceMTEB/FinTruthQA",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "question",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/FinQA",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "question",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/TATQA",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "question",
            "weight": 0.010,
        },
        {
            "name": "FinanceMTEB/synthetic_pii_finance_en",
            "config": None,
            "split": "test",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.006,
        },
        {
            "name": "sujet-ai/Sujet-Finance-Instruct-177k",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "user_prompt",
            "weight": 0.18,
        },
        {
            "name": "FinGPT/fingpt-sentiment-train",
            "config": None,
            "split": "train",
            "text_field": "input",
            "instruction_field": None,
            "label_field": "output",
            "label_map": None,
            "prompt_template": "What is the sentiment of this news? Answer with: negative, neutral, or positive.\n\nNews:\n{text}",
            "weight": 0.04,
        },
        {
            "name": "nickmuchi/financial-classification",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "labels",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of this financial text.\n\nText:\n{text}",
            "weight": 0.02,
        },
        # High-quality general instruction-following — added v6.0.0 for format consistency
        {
            "name": "yahma/alpaca-cleaned",
            "config": None,
            "split": "train",
            "text_field": "output",
            "instruction_field": "instruction",
            "weight": 0.05,
        },
    ]

    LIGHT_DATASETS = [
        {
            "name": "gbharti/finance-alpaca",
            "config": None,
            "split": "train",
            "text_field": "output",
            "instruction_field": "instruction",
            "weight": 0.40,
        },
        {
            "name": "yahma/alpaca-cleaned",
            "config": None,
            "split": "train",
            "text_field": "output",
            "instruction_field": "instruction",
            "weight": 0.20,
        },
        {
            "name": "FinanceMTEB/financial_phrasebank",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following financial sentence.\n\nSentence:\n{text}",
            "weight": 0.06,
        },
        {
            "name": "FinanceMTEB/FinSent",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following sentence.\n\nSentence:\n{text}",
            "weight": 0.06,
        },
        {
            "name": "FinanceMTEB/OpenFinDataSentiment",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following financial text.\n\nText:\n{text}",
            "weight": 0.06,
        },
        {
            "name": "FinanceMTEB/FiQA_ABSA",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) expressed in the following finance-related text.\n\nText:\n{text}",
            "weight": 0.05,
        },
        {
            "name": "FinanceMTEB/SemEva2017_Headline",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following headline.\n\nHeadline:\n{text}",
            "weight": 0.05,
        },
        {
            "name": "FinanceMTEB/FinancialFraud",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "prompt_template": "Determine whether the following case description indicates potential financial fraud. Answer with a short label.\n\nText:\n{text}",
            "weight": 0.05,
        },
        {
            "name": "FinanceMTEB/Complaints",
            "config": None,
            "split": "test",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "prompt_template": "Classify the product/category for the following financial complaint.\n\nComplaint:\n{text}",
            "weight": 0.05,
        },
        {
            "name": "FinanceMTEB/FOMC",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/ESG",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "prompt_template": "Given the following text, classify it into the appropriate ESG-related category.\n\nText:\n{text}",
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/FLS",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/FinFE",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/FinEvaSentiment",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following text.\n\nText:\n{text}",
            "weight": 0.05,
        },
        {
            "name": "FinanceMTEB/FinChinaSentiment",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "label",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of the following text.\n\nText:\n{text}",
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/TradeTheEventNews",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.03,
        },
        {
            "name": "FinanceMTEB/TradeTheEventEncyclopedia",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.03,
        },
        {
            "name": "FinanceMTEB/AlphaFin",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.03,
        },
        {
            "name": "FinanceMTEB/FinTruthQA",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "question",
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/FinQA",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "question",
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/TATQA",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "question",
            "weight": 0.04,
        },
        {
            "name": "FinanceMTEB/synthetic_pii_finance_en",
            "config": None,
            "split": "test",
            "text_field": "text",
            "instruction_field": None,
            "weight": 0.02,
        },
        {
            "name": "sujet-ai/Sujet-Finance-Instruct-177k",
            "config": None,
            "split": "train",
            "text_field": "answer",
            "instruction_field": "user_prompt",
            "weight": 0.12,
        },
        {
            "name": "FinGPT/fingpt-sentiment-train",
            "config": None,
            "split": "train",
            "text_field": "input",
            "instruction_field": None,
            "label_field": "output",
            "label_map": None,
            "prompt_template": "What is the sentiment of this news? Answer with: negative, neutral, or positive.\n\nNews:\n{text}",
            "weight": 0.04,
        },
        {
            "name": "nickmuchi/financial-classification",
            "config": None,
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": "labels",
            "label_map": {0: "negative", 1: "neutral", 2: "positive"},
            "prompt_template": "Classify the sentiment (negative/neutral/positive) of this financial text.\n\nText:\n{text}",
            "weight": 0.02,
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
        # Derive a shuffle seed from skip_items so each "epoch" sees different data.
        # This replaces the old sequential .skip() approach which required downloading
        # and discarding thousands of streaming items (70+ min overhead per run).
        self.shuffle_seed = skip_items % 100_000
        self.max_bytes_per_run = max_bytes_per_run
        # Streaming-shuffle look-ahead buffer PER dataset. With ~25 datasets opened at
        # once, a large buffer multiplies: each one pulls that many items (and their
        # backing Parquet row-groups) into Arrow memory, so buffer_size=2000 grew RSS by
        # ~4.4GB after only 80 items and OOM-killed the 16GB CPU runner. Keep it small.
        self.shuffle_buffer = max(1, int(os.getenv("SHUFFLE_BUFFER", "128")))
        # A few datasets (fineweb-edu, OpenMathInstruct-2) have multi-GB Parquet
        # row-groups. Streaming all of them at once was the dominant RAM term that
        # filled the 16GB runner. Cap how many "heavy" datasets stream concurrently;
        # which heavy ones are active rotates per run (seeded by shuffle_seed), so the
        # full curriculum is still covered across hourly runs. 0 = no cap.
        self.max_heavy_concurrent = int(os.getenv("MAX_HEAVY_CONCURRENT", "1"))
        self.items_processed = 0
        self.datasets = (
            self.LIGHT_DATASETS if int(os.getenv("USE_LIGHT_DATASETS", "0")) == 1 else self.DATASETS
        )

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
                    print(f"  [FAIL] Failed to load {ds_config['name']}: {e}")
                    return None

    def _format_text(self, item: dict, ds_config: dict) -> str:
        """Format a dataset item into training text.

        For instruction datasets, creates instruction-response format.
        For plain text datasets, uses raw text.
        """
        instruction = ""
        if ds_config.get("instruction_field") and ds_config["instruction_field"] in item:
            instruction = item[ds_config["instruction_field"]]

        text = item.get(ds_config.get("text_field", ""), "")
        if not isinstance(text, str):
            text = str(text) if text else ""

        label_field = ds_config.get("label_field")
        label_map = ds_config.get("label_map")
        prompt_template = ds_config.get("prompt_template")

        label_value = None
        if label_field and label_field in item:
            label_value = item[label_field]
            if label_map and isinstance(label_value, int) and label_value in label_map:
                label_value = label_map[label_value]

        eos = self.tokenizer.eos_token or ""

        if prompt_template and text:
            instr = prompt_template.format(text=text)
            if label_value is not None and label_value != "":
                return f"### Instruction:\n{instr}\n\n### Response:\n{label_value}{eos}"
            return f"### Instruction:\n{instr}\n\n### Response:\n{eos}"

        if instruction and text:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{text}{eos}"
        if text:
            return f"{text}{eos}"
        return ""

    def _select_datasets(self) -> list:
        """Return the datasets to stream this run, capping concurrent heavy ones.

        Heavy datasets have multi-GB Parquet row-groups; streaming all of them at
        once exhausts the 16GB CPU runner. We keep every light dataset and admit at
        most ``max_heavy_concurrent`` heavy ones, rotating which by a per-run seed so
        all heavy data is still seen across hourly runs.
        """
        heavy = [d for d in self.datasets if d.get("heavy")]
        light = [d for d in self.datasets if not d.get("heavy")]
        if self.max_heavy_concurrent <= 0 or len(heavy) <= self.max_heavy_concurrent:
            return self.datasets
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(heavy)
        selected_heavy = heavy[: self.max_heavy_concurrent]
        if selected_heavy:
            print(
                "  [INFO] Heavy-dataset cap: streaming "
                f"{[d['name'] for d in selected_heavy]} this run (of {len(heavy)} heavy)"
            )
        # Preserve original ordering for the rest of the pipeline's weighted round-robin.
        keep = set(id(d) for d in light) | set(id(d) for d in selected_heavy)
        return [d for d in self.datasets if id(d) in keep]

    def stream(self) -> Iterator[dict]:
        """Yield tokenized examples from mixed datasets."""
        # Load all dataset streams
        streams = []
        for ds_config in self._select_datasets():
            dataset = self._load_stream(ds_config)
            if dataset is not None:
                # Shuffle with a seed derived from the run's position counter.
                # This gives each hourly run a different data sample without the
                # O(skip_items) sequential download overhead of dataset.skip().
                # buffer_size is kept small (see __init__) because ~25 concurrent streams
                # multiply it into multi-GB Arrow row-group memory on the CPU runner.
                try:
                    dataset = dataset.shuffle(
                        seed=self.shuffle_seed, buffer_size=self.shuffle_buffer
                    )
                except Exception:
                    pass  # Some datasets may not support shuffle; use as-is
                streams.append((iter(dataset), ds_config))
                print(f"  [OK] Loaded {ds_config['name']} (weight: {ds_config['weight']})")

        if not streams:
            print("  [FAIL] No datasets loaded! Falling back to synthetic data.")
            yield from self._synthetic_fallback()
            return

        total_bytes = 0
        # Round-robin with weights
        stream_indices = list(range(len(streams)))
        weights = [streams[i][1]["weight"] for i in range(len(streams))]

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
                padding=False,
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].squeeze(0)

            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = 0

            # Pad only up to the nearest multiple of 8 (less waste than max_length padding)
            seq_len = int(input_ids.size(0))
            pad_to = min(self.block_size, ((seq_len + 7) // 8) * 8)
            if seq_len < pad_to:
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_to - seq_len), value=pad_id)

            attention_mask = (input_ids != pad_id).long()
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
    max_bytes: int = 15 * 1024 * 1024,
) -> torch.utils.data.DataLoader:
    """Create the finance-focused training DataLoader."""
    pipeline = FinanceDataPipeline(
        tokenizer=tokenizer,
        block_size=block_size,
        skip_items=skip_items,
        max_bytes_per_run=max_bytes,
    )

    dataset = _IterableDataset(pipeline.stream)

    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        # Find max length in this batch (cap to block_size, pad to multiple of 8)
        max_len = 0
        for ex in batch:
            length = int(ex["input_ids"].numel())
            if length > max_len:
                max_len = length
        max_len = min(int(block_size), max_len)
        max_len = max(1, ((max_len + 7) // 8) * 8)
        max_len = min(int(block_size), max_len)

        input_ids_out = []
        attn_out = []
        labels_out = []
        processed_idx_out = []
        for ex in batch:
            ids = ex["input_ids"][:max_len]
            labels = ex["labels"][:max_len]
            pad_amt = max_len - int(ids.numel())
            if pad_amt > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_amt), value=pad_id)
                labels = torch.nn.functional.pad(labels, (0, pad_amt), value=-100)

            attn = (ids != pad_id).long()

            input_ids_out.append(ids)
            labels_out.append(labels)
            attn_out.append(attn)
            if "processed_idx" in ex:
                processed_idx_out.append(ex["processed_idx"])

        out = {
            "input_ids": torch.stack(input_ids_out, dim=0),
            "attention_mask": torch.stack(attn_out, dim=0),
            "labels": torch.stack(labels_out, dim=0),
        }
        if processed_idx_out:
            out["processed_idx"] = torch.tensor(processed_idx_out, dtype=torch.long)
        return out

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )


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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
