#!/usr/bin/env python3
"""
Quick training script for debugging - trains on small data slice
"""

import multiprocessing
import os

import torch
from transformers import AutoTokenizer

from fin_ai.data.dataset import create_dataloader, load_datasets_from_config
from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.training.trainer import DatasetCycler, FinAITrainer, TrainingConfig


def main():
    # CPU Optimization
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    print(f"Optimizing for CPU: using {num_cores} threads")

    # Load configuration
    config = FinAIConfig()
    train_config = TrainingConfig.from_yaml("config/model_config.yaml")
    
    # Override for quick training
    train_config.max_steps = 50  # Just 50 steps for quick test
    train_config.save_steps = 25  # Save halfway through
    train_config.log_steps = 5
    train_config.batch_size = 1
    train_config.gradient_accumulation_steps = 4  # Smaller for faster iteration
    
    print(f"Quick training mode: {train_config.max_steps} steps")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = ["<TICKER>", "<ACCOUNTING>", "<SEC_FILING>", "<ARXIV_FIN>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.vocab_size = len(tokenizer)

    # Initialize Model
    model_path = "checkpoints/model"
    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        print(f"Loading existing model from {model_path}")
        try:
            model = FinAIForCausalLM.from_pretrained(model_path)
            print(f"✓ Loaded model with {model.num_parameters():,} parameters")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Initializing new model from scratch.")
            model = FinAIForCausalLM(config)
    else:
        print("No local model found. Initializing new model from scratch.")
        model = FinAIForCausalLM(config)

    # Initialize Dataset Cycler
    cycler = DatasetCycler("config/datasets.yaml")
    current_offset = cycler.get_current_offset()

    # Load small dataset slice - approximately 10-25MB
    # With 512 token sequences, ~200 samples = ~10-15MB of text
    print("Loading small dataset slice (200 samples, ~10-15MB)...")
    dataset, next_offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_samples=200,  # Small slice for quick training
        offset=current_offset,
    )

    cycler.increment_offset(next_offset - current_offset)

    dataloader = create_dataloader(dataset, batch_size=train_config.batch_size)

    print(f"\n{'='*60}")
    print(f"Starting Quick Training Run")
    print(f"{'='*60}")
    print(f"Model: {model.num_parameters():,} parameters")
    print(f"Dataset: {cycler.current_dataset_name}")
    print(f"Offset: {current_offset} -> {next_offset}")
    print(f"Samples: 200 (~10-15MB)")
    print(f"Steps: {train_config.max_steps}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"{'='*60}\n")

    trainer = FinAITrainer(
        model=model,
        train_dataloader=dataloader,
        config=train_config,
        dataset_cycler=cycler,
    )

    trainer.train()

    # Save final model
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path, safe_serialization=False)
    tokenizer.save_pretrained(model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Push to Hugging Face
    hf_token = trainer._get_hf_token()
    if hf_token and train_config.hf_repo_id:
        from huggingface_hub import HfApi, create_repo

        try:
            print(f"\nPushing to Hugging Face: {train_config.hf_repo_id}")

            # Create repo if it doesn't exist
            create_repo(
                repo_id=train_config.hf_repo_id,
                token=hf_token,
                private=True,
                exist_ok=True,
            )

            # Upload
            api = HfApi(token=hf_token)
            api.upload_folder(
                folder_path=model_path,
                repo_id=train_config.hf_repo_id,
                commit_message=f"Quick training - offset {next_offset} - {train_config.max_steps} steps",
            )

            print("✓ Push to Hugging Face successful")
            print(f"✓ Model available at: https://huggingface.co/{train_config.hf_repo_id}")
        except Exception as e:
            print(f"✗ Failed to push to Hugging Face: {e}")
            print("Model saved locally, will retry on next run")
    else:
        print("\n⚠ No HF_TOKEN found, skipping push to Hugging Face")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
