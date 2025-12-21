#!/usr/bin/env python3
"""
Fin.AI Training Script

Usage:
    python train.py --config config/model_config.yaml --datasets config/datasets.yaml
"""

import argparse
import logging
import os
import sys

import torch
from transformers import AutoTokenizer

from fin_ai.model import FinAIModel, FinAIConfig
from fin_ai.data import load_datasets_from_config, create_dataloader
from fin_ai.training import FinAITrainer, TrainingConfig, DatasetCycler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()
    
    # Load configs
    logger.info("Loading configurations...")
    model_config = FinAIConfig.from_yaml(args.config)
    training_config = TrainingConfig.from_yaml(args.config)
    
    # Apply overrides
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.max_steps:
        training_config.max_steps = args.max_steps
    
    # Log model info
    logger.info(f"Model config: {model_config.to_dict()}")
    logger.info(f"Estimated parameters: {model_config.num_parameters:,}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config.vocab_size = len(tokenizer)
    
    # Initialize dataset cycler
    dataset_cycler = DatasetCycler(
        args.datasets,
        state_file=os.path.join(training_config.output_dir, "dataset_state.json")
    )
    
    logger.info(f"Starting with dataset: {dataset_cycler.current_dataset_name}")
    
    # Load datasets
    logger.info("Loading datasets...")
    dataset = load_datasets_from_config(
        args.datasets,
        tokenizer=tokenizer,
        max_seq_len=model_config.max_seq_len,
        max_samples=args.max_samples,
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    # Create model
    logger.info("Creating model...")
    model = FinAIModel(model_config)
    
    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = FinAITrainer(
        model=model,
        train_dataloader=train_dataloader,
        config=training_config,
        dataset_cycler=dataset_cycler,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
