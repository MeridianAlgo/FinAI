#!/usr/bin/env python3
"""Quick 50-step training test"""

import multiprocessing
import os
import time

import torch
import yaml
from transformers import AutoTokenizer

from fin_ai.data.dataset import create_dataloader, load_datasets_from_config
from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.training.trainer import DatasetCycler, FinAITrainer, TrainingConfig


def main():
    print("=" * 60, flush=True)
    print("Quick 50-Step Training Test", flush=True)
    print("=" * 60, flush=True)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("", flush=True)

    # CPU Optimization
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)

    # Load configuration
    print("Loading training config...", flush=True)
    train_config = TrainingConfig.from_yaml("config/model_config.yaml")

    # Create model config from YAML
    print("Loading model config from YAML...", flush=True)
    with open("config/model_config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    model_config_dict = yaml_config.get("model", {})
    config = FinAIConfig(**model_config_dict)

    print("Training config:", flush=True)
    print(f"  - Max steps: {train_config.max_steps}", flush=True)
    print(f"  - Batch size: {train_config.batch_size}", flush=True)
    print(
        f"  - Gradient accumulation: {train_config.gradient_accumulation_steps}",
        flush=True,
    )
    print("", flush=True)

    # Initialize Tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Adding special tokens...", flush=True)
    special_tokens = ["<TICKER>", "<ACCOUNTING>", "<SEC_FILING>", "<ARXIV_FIN>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.vocab_size = len(tokenizer)
    print(f"Tokenizer ready with vocab size {len(tokenizer)}", flush=True)

    # Load or initialize model
    model_path = "checkpoints/model"
    print(f"Checking model path: {model_path}", flush=True)
    print(f"Model path exists: {os.path.exists(model_path)}", flush=True)
    if os.path.exists(model_path):
        print(f"Model path contents: {os.listdir(model_path)}", flush=True)

    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        print(f"Loading model from {model_path}...", flush=True)
        start = time.time()
        try:
            model = FinAIForCausalLM.from_pretrained(model_path)
            print(f"Model loaded in {time.time() - start:.1f}s", flush=True)
        except Exception as e:
            print(f"Failed to load: {e}", flush=True)
            print("Initializing new model...", flush=True)
            model = FinAIForCausalLM(config)
    else:
        print("Initializing new model...", flush=True)
        start = time.time()
        model = FinAIForCausalLM(config)
        print(f"Model initialized in {time.time() - start:.1f}s", flush=True)

    print(f"Model parameters: {model.num_parameters():,}", flush=True)
    print("", flush=True)

    # Load dataset
    print("Creating dataset cycler...", flush=True)
    cycler = DatasetCycler("config/datasets.yaml")
    current_offset = cycler.get_current_offset()

    print(f"Loading dataset (50 samples from offset {current_offset})...", flush=True)
    start = time.time()
    dataset, next_offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_samples=50,  # Reduced for faster test
        offset=current_offset,
    )
    print(f"Dataset loaded in {time.time() - start:.1f}s", flush=True)
    print("", flush=True)

    print("Incrementing offset...", flush=True)
    cycler.increment_offset(next_offset - current_offset)
    print("Creating dataloader...", flush=True)
    dataloader = create_dataloader(dataset, batch_size=train_config.batch_size)
    print("Dataloader created", flush=True)

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
    print(f"Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} minutes)")
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
