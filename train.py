#!/usr/bin/env python3
"""
FinAI-Core v2.2 Ultra-Lite Training Script
Optimized for Continual Learning and CPU Performance
"""

import multiprocessing
import os

import torch
import yaml
from transformers import AutoTokenizer

from fin_ai.data.dataset import create_dataloader, load_datasets_from_config
from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.training.trainer import DatasetCycler, FinAITrainer, TrainingConfig


def main():
    # CPU Optimization
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    print("========================================")
    print("FinAI-Core v2.2 Training")
    print("========================================")
    print(f"Timestamp: {__import__('datetime').datetime.now()}")
    print(f"CPU cores: {num_cores}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("")

    # Load configuration
    print("Loading configuration...")
    train_config = TrainingConfig.from_yaml("config/model_config.yaml")
    with open("config/model_config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)
    config = FinAIConfig(**yaml_config.get("model", {}))
    print("[OK] Config loaded")
    print(f"  - Batch size: {train_config.batch_size}")
    print(f"  - Max steps: {train_config.max_steps}")
    print(f"  - Learning rate: {train_config.learning_rate}")
    print("")

    # Initialize Tokenizer (gpt2 base + finance tokens)
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Add special finance tokens
    special_tokens = ["<TICKER>", "<ACCOUNTING>", "<SEC_FILING>", "<ARXIV_FIN>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    config.vocab_size = len(tokenizer)
    print(f"[OK] Tokenizer initialized (vocab size: {config.vocab_size})")
    print("")

    # Initialize Model
    model_path = "checkpoints/model"
    print(f"Checking for existing model at {model_path}...")
    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        print("[OK] Found existing model, attempting to load...")
        print(f"  Files in {model_path}:")
        for f in os.listdir(model_path)[:10]:
            size = os.path.getsize(os.path.join(model_path, f)) / 1024 / 1024
            print(f"    - {f}: {size:.1f} MB")
        try:
            import time

            start = time.time()
            model = FinAIForCausalLM.from_pretrained(model_path)
            elapsed = time.time() - start
            print(f"[OK] Model loaded successfully in {elapsed:.1f}s")
        except Exception as e:
            print(f"[FAIL] Failed to load model: {e}")
            print(f"  Error type: {type(e).__name__}")
            print("Initializing new model from scratch.")
            model = FinAIForCausalLM(config)
    else:
        print("[FAIL] No local model found. Initializing new model from scratch.")
        model = FinAIForCausalLM(config)

    print(f"Model parameters: {model.num_parameters():,}")
    
    # Detailed Model Diagnostics
    print("========================================")
    print("Model Diagnostics")
    print("========================================")
    
    # 1. Parameter Breakdown
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    layer_params = sum(p.numel() for p in model.model.layers.parameters())
    
    print(f"  - Embeddings: {embed_params:,}")
    print(f"  - Layers:     {layer_params:,}")
    print(f"  - LM Head:    {lm_head_params:,}")
    
    # 2. Weight Tying Check
    is_tied = model.lm_head.weight is model.model.embed_tokens.weight
    print(f"  - Weight Tying: {'[OK] Tied' if is_tied else '[WARN] Not Tied'}")
    
    # 3. Health Check (NaN/Inf)
    print("  - Checking weights for NaNs/Infs...")
    has_issue = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"    [FAIL] NaN found in {name}")
            has_issue = True
        if torch.isinf(param).any():
            print(f"    [FAIL] Inf found in {name}")
            has_issue = True
    
    if not has_issue:
        print("    [OK] Weights are healthy (no NaN/Inf)")
    else:
        print("    [CRITICAL] Model initialized with bad weights!")
        if not os.environ.get("GITHUB_ACTIONS"):
            input("Press Enter to continue anyway or Ctrl+C to stop...")
    
    print("========================================")
    print("")

    # Initialize Dataset Cycler to track offsets
    print("Initializing dataset cycler...")
    cycler = DatasetCycler("config/datasets.yaml")
    current_offset = cycler.get_current_offset()
    print("[OK] Dataset cycler initialized")
    print(f"  - Current dataset: {cycler.current_dataset_name}")
    print(f"  - Current offset: {current_offset}")
    print("")

    # Load dataset with current offset
    print(f"Loading dataset (max 1000 samples from offset {current_offset})...")
    import time

    start = time.time()
    dataset, next_offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_samples=1000,  # Reduced from 5000 for faster CI runs
        offset=current_offset,
    )
    elapsed = time.time() - start
    print(f"[OK] Dataset loaded in {elapsed:.1f}s")
    print(f"  - Samples loaded: {next_offset - current_offset}")
    print(f"  - New offset: {next_offset}")
    print("")

    # Update cycler with how many samples we actually skipped/read
    cycler.increment_offset(next_offset - current_offset)

    dataloader = create_dataloader(dataset, batch_size=train_config.batch_size)
    print("[OK] Dataloader created")
    print("")

    print("========================================")
    print("Starting Training")
    print("========================================")
    print(f"Dataset: {cycler.current_dataset_name}")
    print(f"Offset: {current_offset} -> {next_offset}")
    print(f"Model: {model.num_parameters():,} parameters")
    print(f"Steps: {train_config.max_steps}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print("")

    trainer = FinAITrainer(
        model=model,
        train_dataloader=dataloader,
        config=train_config,
        dataset_cycler=cycler,
    )

    trainer.train()

    # Save final model
    print("")
    print(f"Saving model to {model_path}...")
    os.makedirs(model_path, exist_ok=True)
    start = time.time()
    model.save_pretrained(model_path, safe_serialization=False)
    tokenizer.save_pretrained(model_path)
    elapsed = time.time() - start
    print(f"[OK] Model saved in {elapsed:.1f}s")
    print("")

    # Push to Hugging Face if HF_TOKEN is available
    hf_token = trainer._get_hf_token()
    if hf_token and train_config.hf_repo_id:
        from huggingface_hub import HfApi, create_repo

        try:
            print("========================================")
            print("Pushing to Hugging Face")
            print("========================================")
            print(f"Repository: {train_config.hf_repo_id}")
            print("")

            # Create repo if it doesn't exist
            print("Creating/verifying repository...")
            create_repo(
                repo_id=train_config.hf_repo_id,
                token=hf_token,
                private=True,
                exist_ok=True,
            )
            print("[OK] Repository ready")
            print("")

            # Use HfApi for more efficient uploads
            api = HfApi(token=hf_token)

            # Get model size
            total_size = sum(
                os.path.getsize(os.path.join(model_path, f))
                for f in os.listdir(model_path)
                if os.path.isfile(os.path.join(model_path, f))
            )
            print(f"Uploading {total_size / 1024 / 1024 / 1024:.2f} GB...")
            print("This may take several minutes...")
            print("")

            # Upload folder with resume capability
            start = time.time()
            api.upload_folder(
                folder_path=model_path,
                repo_id=train_config.hf_repo_id,
                commit_message=f"Train cycle complete - offset {next_offset}",
            )
            elapsed = time.time() - start

            print("")
            print(f"[OK] Push to Hugging Face successful in {elapsed:.1f}s")
            print(
                f"[OK] Model available at: https://huggingface.co/{train_config.hf_repo_id}"
            )
        except Exception as e:
            print("")
            print(f"[FAIL] Failed to push to Hugging Face: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            print("Model saved locally, will retry on next run")
    else:
        print("[WARN] No HF_TOKEN or repo_id configured, skipping push")

    print("")
    print("========================================")
    print("Training Complete!")
    print("========================================")
    print(f"Timestamp: {__import__('datetime').datetime.now()}")


if __name__ == "__main__":
    main()
