#!/usr/bin/env python3
"""Quick 50-step training test"""

import multiprocessing
import os
import time

import torch
from transformers import AutoTokenizer

from fin_ai.data.dataset import create_dataloader, load_datasets_from_config
from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.training.trainer import DatasetCycler, FinAITrainer, TrainingConfig


def main():
    print("=" * 60)
    print("Quick 50-Step Training Test")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # CPU Optimization
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)

    # Load configuration
    config = FinAIConfig()
    train_config = TrainingConfig.from_yaml("config/model_config.yaml")

    print(f"Training config:")
    print(f"  - Max steps: {train_config.max_steps}")
    print(f"  - Batch size: {train_config.batch_size}")
    print(f"  - Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print("")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = ["<TICKER>", "<ACCOUNTING>", "<SEC_FILING>", "<ARXIV_FIN>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.vocab_size = len(tokenizer)

    # Load or initialize model
    model_path = "checkpoints/model"
    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        print(f"Loading model from {model_path}...")
        start = time.time()
        try:
            model = FinAIForCausalLM.from_pretrained(model_path)
            print(f"Model loaded in {time.time() - start:.1f}s")
        except Exception as e:
            print(f"Failed to load: {e}")
            print("Initializing new model...")
            model = FinAIForCausalLM(config)
    else:
        print("Initializing new model...")
        model = FinAIForCausalLM(config)

    print(f"Model parameters: {model.num_parameters():,}")
    print("")

    # Load dataset
    cycler = DatasetCycler("config/datasets.yaml")
    current_offset = cycler.get_current_offset()

    print(f"Loading dataset (200 samples from offset {current_offset})...")
    start = time.time()
    dataset, next_offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_samples=200,  # Small for quick test
        offset=current_offset,
    )
    print(f"Dataset loaded in {time.time() - start:.1f}s")
    print("")

    cycler.increment_offset(next_offset - current_offset)
    dataloader = create_dataloader(dataset, batch_size=train_config.batch_size)

    # Train
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    start = time.time()

    trainer = FinAITrainer(
        model=model,
        train_dataloader=dataloader,
        config=train_config,
        dataset_cycler=cycler,
    )

    trainer.train()

    elapsed = time.time() - start
    print("")
    print("=" * 60)
    print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print("=" * 60)

    # Save model
    print("")
    print("Saving model...")
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path, safe_serialization=False)
    tokenizer.save_pretrained(model_path)
    print("Model saved!")

    print("")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
