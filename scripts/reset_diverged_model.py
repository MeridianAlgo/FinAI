#!/usr/bin/env python3
"""
Reset a diverged model checkpoint to fresh weights

This script loads the model architecture from a checkpoint but reinitializes
the weights, useful when a model has diverged (NaN loss).

Usage:
    python scripts/reset_diverged_model.py --checkpoint checkpoints/checkpoint-fineweb-edu.pt
"""

import argparse
import os

import torch

from fin_ai.model import FinAIForCausalLM


def reset_checkpoint(checkpoint_path: str, output_path: str = None):
    """Reset a checkpoint to fresh weights while preserving architecture"""

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return False

    if output_path is None:
        output_path = checkpoint_path

    print(f"Loading checkpoint from {checkpoint_path}...")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract config from state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Try to load model to get config
        print("Extracting model configuration...")

        # Load model config from checkpoints/model if available
        model_dir = os.path.join(os.path.dirname(checkpoint_path), "model")
        if os.path.exists(os.path.join(model_dir, "config.json")):
            print(f"Loading model config from {model_dir}")
            model = FinAIForCausalLM.from_pretrained(model_dir)
            config = model.config
        else:
            print(
                "Error: Could not find model config. Please ensure checkpoints/model exists."
            )
            return False

        # Create fresh model with same config
        print("Creating fresh model with same architecture...")
        fresh_model = FinAIForCausalLM(config)

        # Get fresh state dict
        fresh_state_dict = fresh_model.state_dict()

        # Update checkpoint with fresh weights
        checkpoint["model_state_dict"] = fresh_state_dict
        checkpoint["global_step"] = 0
        checkpoint["epoch"] = 0

        # Reset optimizer and scheduler states
        if "optimizer_state_dict" in checkpoint:
            print("Resetting optimizer state...")
            # Keep the structure but reset the states
            for state in checkpoint["optimizer_state_dict"]["state"].values():
                if "step" in state:
                    state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"] = torch.zeros_like(state["exp_avg"])
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"] = torch.zeros_like(state["exp_avg_sq"])

        # Save reset checkpoint
        print(f"Saving reset checkpoint to {output_path}...")
        torch.save(checkpoint, output_path)

        # Also save the model in HF format
        model_save_path = os.path.join(os.path.dirname(output_path), "model")
        print(f"Saving model to {model_save_path}...")

        # Temporarily disable weight tying for save
        original_tie = fresh_model.config.tie_word_embeddings
        fresh_model.config.tie_word_embeddings = False
        fresh_model.save_pretrained(model_save_path, safe_serialization=True)
        fresh_model.config.tie_word_embeddings = original_tie

        print("\n" + "=" * 60)
        print("✓ Successfully reset checkpoint!")
        print("=" * 60)
        print(f"\nCheckpoint: {output_path}")
        print(f"Model: {model_save_path}")
        print("\nThe model now has fresh weights and training will start from step 0.")
        print("=" * 60 + "\n")

        return True

    except Exception as e:
        print(f"Error resetting checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Reset a diverged model checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint-fineweb-edu.pt",
        help="Path to checkpoint file to reset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original checkpoint",
    )

    args = parser.parse_args()

    # Create backup if requested
    if args.backup:
        import shutil

        backup_path = args.checkpoint + ".backup"
        print(f"Creating backup at {backup_path}...")
        shutil.copy2(args.checkpoint, backup_path)
        print("✓ Backup created\n")

    # Reset checkpoint
    success = reset_checkpoint(args.checkpoint, args.output)

    if success:
        print("\nYou can now resume training with the reset checkpoint.")
    else:
        print("\nFailed to reset checkpoint.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
