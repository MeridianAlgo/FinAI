#!/usr/bin/env python3
"""
FinAI-Core v2.2 Ultra-Lite Training Script
Optimized for Continual Learning and CPU Performance
"""

import os
import torch
import multiprocessing
from transformers import AutoTokenizer
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.training.trainer import FinAITrainer, TrainingConfig, DatasetCycler
from fin_ai.data.dataset import load_datasets_from_config, create_dataloader

def main():
    # CPU Optimization
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    print(f"Optimizing for CPU: using {num_cores} threads")

    # Load configuration
    config = FinAIConfig()
    train_config = TrainingConfig.from_yaml("config/model_config.yaml")

    # Initialize Tokenizer (gpt2 base + finance tokens)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Add special finance tokens
    special_tokens = ["<TICKER>", "<ACCOUNTING>", "<SEC_FILING>", "<ARXIV_FIN>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.vocab_size = len(tokenizer)

    # Initialize Model
    model_path = "checkpoints/model"
    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        print(f"Loading existing model from {model_path}")
        model = FinAIForCausalLM.from_pretrained(model_path)
    else:
        # Try to download from Hugging Face
        hf_token = os.environ.get("HF_TOKEN")
        repo_id = train_config.hf_repo_id
        if hf_token and repo_id:
            try:
                print(f"Attempting to download model from HF: {repo_id}")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=model_path, token=hf_token)
                print(f"Downloaded model to {model_path}")
                model = FinAIForCausalLM.from_pretrained(model_path)
            except Exception as e:
                print(f"Could not download from HF ({e}). Initializing new model from scratch.")
                model = FinAIForCausalLM(config)
        else:
            print("No HF_TOKEN or repo_id found. Initializing new model from scratch.")
            model = FinAIForCausalLM(config)

    # Initialize Dataset Cycler to track offsets
    cycler = DatasetCycler("config/datasets.yaml")
    current_offset = cycler.get_current_offset()

    # Load dataset with current offset
    dataset, next_offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_samples=5000, # Train on 5000 samples per run for "slices"
        offset=current_offset
    )

    # Update cycler with how many samples we actually skipped/read
    cycler.increment_offset(next_offset - current_offset)

    dataloader = create_dataloader(dataset, batch_size=train_config.batch_size)

    print(f"Starting FinAI-Core v2.2 training ({model.num_parameters():,} params)")
    print(f"Dataset: {cycler.current_dataset_name}, Offset: {current_offset}")

    trainer = FinAITrainer(
        model=model,
        train_dataloader=dataloader,
        config=train_config,
        dataset_cycler=cycler
    )

    trainer.train()

    # Save final model
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # Push to Hugging Face if HF_TOKEN is available
    hf_token = trainer._get_hf_token()
    if hf_token and train_config.hf_repo_id:
        from huggingface_hub import create_repo, upload_folder
        try:
            print(f"Pushing to Hugging Face: {train_config.hf_repo_id}")
            create_repo(repo_id=train_config.hf_repo_id, token=hf_token, private=True, exist_ok=True)
            upload_folder(
                folder_path=model_path,
                repo_id=train_config.hf_repo_id,
                token=hf_token,
                commit_message=f"Train cycle complete - offset {next_offset}"
            )
            print("Push to Hugging Face successful")
        except Exception as e:
            print(f"Failed to push to Hugging Face: {e}")

if __name__ == "__main__":
    main()
