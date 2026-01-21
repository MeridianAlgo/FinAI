#!/usr/bin/env python3
"""
Fin.AI Training Script

Usage:
    python train.py --config config/model_config.yaml --datasets config/datasets.yaml
"""

import argparse
import gc
import logging
import os
import sys

import torch
from transformers import AutoTokenizer

from fin_ai.data import create_dataloader, load_datasets_from_config
from fin_ai.model import FinAIConfig, FinAIForCausalLM
from fin_ai.training import DatasetCycler, FinAITrainer, TrainingConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress verbose warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cleanup_memory():
    """Aggressive memory cleanup before training"""
    print("Cleaning memory and cache...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Memory cleaned")


def main():
    parser = argparse.ArgumentParser(description="Train Fin.AI model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model/training config",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="config/datasets.yaml",
        help="Path to datasets config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset samples (for testing)",
    )
    parser.add_argument(
        "--size-preset",
        type=str,
        default=None,
        help="Override model size preset (e.g. micro, small, base)",
    )
    args = parser.parse_args()

    # Clean memory before starting
    cleanup_memory()

    # Load configs
    print("Loading configurations...")
    model_config = FinAIConfig.from_yaml(args.config)
    training_config = TrainingConfig.from_yaml(args.config)

    if args.size_preset:
        model_config = FinAIConfig(**{**model_config.to_dict(), "size_preset": args.size_preset})

    # Apply overrides
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", verbose=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config.vocab_size = len(tokenizer)

    # CPU-friendly defaults
    if not torch.cuda.is_available():
        model_config.use_flash_attention = False

    # Initialize dataset cycler
    dataset_cycler = DatasetCycler(
        args.datasets,
        state_file=os.path.join(training_config.output_dir, "dataset_state.json"),
    )

    print(f"Dataset: {dataset_cycler.current_dataset_name}")

    # Load datasets
    print("Loading datasets...")
    current_offset = dataset_cycler.get_current_offset()

    dataset, new_offset = load_datasets_from_config(
        args.datasets,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        max_samples=args.max_samples,
        offset=current_offset,
    )

    # Create dataloaders
    train_dataloader = create_dataloader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else 4,  # Windows compatibility
    )

    # Create model
    print("Creating model...")

    model_dir = os.path.join(training_config.output_dir, "model")
    if os.path.exists(os.path.join(model_dir, "config.json")):
        try:
            print(f"Loading model from {model_dir}...")
            model = FinAIForCausalLM.from_pretrained(model_dir)
            model_config = model.config
        except UnicodeEncodeError:
            print(f"Loading model from {model_dir}...")
            model = FinAIForCausalLM.from_pretrained(model_dir)
            model_config = model.config
    else:
        model = FinAIForCausalLM(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready: {total_params:,} parameters")

    # Create trainer
    trainer = FinAITrainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config,
        dataset_cycler=dataset_cycler,
    )

    # Train
    trainer.train()

    # Update dataset state
    if new_offset > current_offset:
        print(f"Updating dataset offset: {current_offset} -> {new_offset}")
        dataset_cycler.dataset_offsets[dataset_cycler.current_dataset_name] = new_offset
        dataset_cycler._save_state()


if __name__ == "__main__":
    main()
