#!/usr/bin/env python3
"""
Simple test to see where training hangs
"""

import multiprocessing
import os
import time

import torch
from transformers import AutoTokenizer

from fin_ai.data.dataset import create_dataloader, load_datasets_from_config
from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.training.trainer import DatasetCycler, FinAITrainer, TrainingConfig

print("=" * 60)
print("SIMPLE TRAINING TEST")
print("=" * 60)
print()

# CPU Optimization
num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
print(f"[{time.strftime('%H:%M:%S')}] CPU cores: {num_cores}")
print(f"[{time.strftime('%H:%M:%S')}] PyTorch version: {torch.__version__}")
print()

# Load configuration
print(f"[{time.strftime('%H:%M:%S')}] Loading configuration...")
config = FinAIConfig()
train_config = TrainingConfig.from_yaml("config/model_config.yaml")
print(
    f"[{time.strftime('%H:%M:%S')}] Config loaded - max_steps: {train_config.max_steps}"
)
print()

# Initialize Tokenizer
print(f"[{time.strftime('%H:%M:%S')}] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_tokens = ["<TICKER>", "<ACCOUNTING>", "<SEC_FILING>", "<ARXIV_FIN>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
config.vocab_size = len(tokenizer)
print(
    f"[{time.strftime('%H:%M:%S')}] Tokenizer ready - vocab size: {config.vocab_size}"
)
print()

# Initialize Model
model_path = "checkpoints/model"
print(f"[{time.strftime('%H:%M:%S')}] Checking for model at {model_path}...")
if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
    print(f"[{time.strftime('%H:%M:%S')}] Found existing model, loading...")
    try:
        start = time.time()
        model = FinAIForCausalLM.from_pretrained(model_path)
        elapsed = time.time() - start
        print(
            f"[{time.strftime('%H:%M:%S')}] Model loaded in {elapsed:.1f}s - {model.num_parameters():,} params"
        )
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Failed to load: {e}")
        print(f"[{time.strftime('%H:%M:%S')}] Initializing new model...")
        model = FinAIForCausalLM(config)
else:
    print(f"[{time.strftime('%H:%M:%S')}] No model found, initializing new...")
    model = FinAIForCausalLM(config)
print()

# Initialize Dataset Cycler
print(f"[{time.strftime('%H:%M:%S')}] Initializing dataset cycler...")
cycler = DatasetCycler("config/datasets.yaml")
current_offset = cycler.get_current_offset()
print(
    f"[{time.strftime('%H:%M:%S')}] Dataset: {cycler.current_dataset_name}, Offset: {current_offset}"
)
print()

# Load dataset - THIS IS WHERE IT LIKELY HANGS
print(
    f"[{time.strftime('%H:%M:%S')}] Loading dataset (max 1000 samples from offset {current_offset})..."
)
print(
    f"[{time.strftime('%H:%M:%S')}] This may take a while if streaming from HuggingFace..."
)
print()

start = time.time()
try:
    dataset, next_offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_samples=1000,
        offset=current_offset,
    )
    elapsed = time.time() - start
    print()
    print(f"[{time.strftime('%H:%M:%S')}] Dataset loaded in {elapsed:.1f}s")
    print(
        f"[{time.strftime('%H:%M:%S')}] Samples: {next_offset - current_offset}, New offset: {next_offset}"
    )
    print()
except Exception as e:
    print()
    print(f"[{time.strftime('%H:%M:%S')}] ERROR loading dataset: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Update cycler
cycler.increment_offset(next_offset - current_offset)

# Create dataloader
print(f"[{time.strftime('%H:%M:%S')}] Creating dataloader...")
dataloader = create_dataloader(dataset, batch_size=train_config.batch_size)
print(f"[{time.strftime('%H:%M:%S')}] Dataloader ready")
print()

# Create trainer
print(f"[{time.strftime('%H:%M:%S')}] Creating trainer...")
trainer = FinAITrainer(
    model=model,
    train_dataloader=dataloader,
    config=train_config,
    dataset_cycler=cycler,
)
print(f"[{time.strftime('%H:%M:%S')}] Trainer ready")
print()

# Train
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"[{time.strftime('%H:%M:%S')}] Steps: {train_config.max_steps}")
print(f"[{time.strftime('%H:%M:%S')}] Batch size: {train_config.batch_size}")
print()

start = time.time()
trainer.train()
elapsed = time.time() - start

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(
    f"[{time.strftime('%H:%M:%S')}] Training took {elapsed:.1f}s ({elapsed / 60:.1f} minutes)"
)
print()

# Save model
print(f"[{time.strftime('%H:%M:%S')}] Saving model...")
os.makedirs(model_path, exist_ok=True)
start = time.time()
model.save_pretrained(model_path, safe_serialization=False)
tokenizer.save_pretrained(model_path)
elapsed = time.time() - start
print(f"[{time.strftime('%H:%M:%S')}] Model saved in {elapsed:.1f}s")
print()

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
