#!/usr/bin/env python3
"""
Check the health of a model checkpoint

This script analyzes a checkpoint and reports:
- Whether the model produces NaN/Inf outputs
- Model statistics (parameter count, norms, etc.)
- Training progress (step, epoch, etc.)

Usage:
    python scripts/check_checkpoint_health.py --checkpoint checkpoints/checkpoint-fineweb-edu.pt
"""

import argparse
import os
import sys

import torch
from transformers import AutoTokenizer


def check_checkpoint_health(checkpoint_path: str):
    """Check if a checkpoint is healthy or diverged"""

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return False

    print("\n" + "=" * 70)
    print(f"CHECKPOINT HEALTH CHECK: {os.path.basename(checkpoint_path)}")
    print("=" * 70 + "\n")

    try:
        # Load checkpoint
        print("📂 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Check structure
        if isinstance(checkpoint, dict):
            print("✓ Checkpoint is a dictionary")

            # Print training info
            if "global_step" in checkpoint:
                print(f"  • Global step: {checkpoint['global_step']:,}")
            if "epoch" in checkpoint:
                print(f"  • Epoch: {checkpoint['epoch']}")
            if "dataset" in checkpoint:
                print(f"  • Dataset: {checkpoint['dataset']}")

            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            print("ℹ️  Checkpoint is a state dict")
            state_dict = checkpoint

        # Check for NaN/Inf in weights
        print("\n🔍 Checking model weights...")
        nan_params = []
        inf_params = []
        zero_params = []
        very_large_params = []

        total_params = 0
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                total_params += param.numel()

                if torch.isnan(param).any():
                    nan_params.append(name)
                if torch.isinf(param).any():
                    inf_params.append(name)
                if (param == 0).all():
                    zero_params.append(name)
                if param.abs().max() > 100:
                    very_large_params.append((name, param.abs().max().item()))

        print(f"  • Total parameters: {total_params:,}")

        if nan_params:
            print(f"\n❌ Found NaN in {len(nan_params)} parameter(s):")
            for name in nan_params[:5]:  # Show first 5
                print(f"     - {name}")
            if len(nan_params) > 5:
                print(f"     ... and {len(nan_params) - 5} more")
        else:
            print("  ✓ No NaN values found")

        if inf_params:
            print(f"\n❌ Found Inf in {len(inf_params)} parameter(s):")
            for name in inf_params[:5]:
                print(f"     - {name}")
            if len(inf_params) > 5:
                print(f"     ... and {len(inf_params) - 5} more")
        else:
            print("  ✓ No Inf values found")

        if very_large_params:
            print(
                f"\n⚠️  Found very large values in {len(very_large_params)} parameter(s):"
            )
            for name, max_val in sorted(
                very_large_params, key=lambda x: x[1], reverse=True
            )[:5]:
                print(f"     - {name}: max={max_val:.2f}")
            if len(very_large_params) > 5:
                print(f"     ... and {len(very_large_params) - 5} more")

        # Load model and test
        print("\n🧪 Testing model inference...")

        # Find model config
        model_dir = os.path.join(os.path.dirname(checkpoint_path), "model")
        if not os.path.exists(os.path.join(model_dir, "config.json")):
            print("⚠️  Could not find model config for inference test")
            is_healthy = len(nan_params) == 0 and len(inf_params) == 0
        else:
            from fin_ai.model import FinAIForCausalLM

            # Load model
            model = FinAIForCausalLM.from_pretrained(model_dir)

            # Load checkpoint weights
            model_state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Remove causal_mask if present
            model_state_dict = {
                k: v for k, v in model_state_dict.items() if "causal_mask" not in k
            }
            model.load_state_dict(model_state_dict, strict=False)
            model.eval()

            # Test with sample input
            print("  • Running inference test...")
            sample_input = torch.randint(0, model.config.vocab_size, (1, 32))

            with torch.no_grad():
                try:
                    outputs = model(input_ids=sample_input, labels=sample_input)
                    loss = outputs.loss
                    logits = outputs.logits

                    if torch.isnan(loss):
                        print("  ❌ Model produces NaN loss")
                        is_healthy = False
                    elif torch.isinf(loss):
                        print("  ❌ Model produces Inf loss")
                        is_healthy = False
                    else:
                        print(
                            f"  ✓ Model inference successful (loss: {loss.item():.4f})"
                        )
                        is_healthy = True

                    if torch.isnan(logits).any():
                        print("  ❌ Model produces NaN logits")
                        is_healthy = False
                    elif torch.isinf(logits).any():
                        print("  ❌ Model produces Inf logits")
                        is_healthy = False
                    else:
                        print(f"  ✓ Logits are healthy (max: {logits.abs().max():.2f})")

                    # Test generation
                    print("  • Testing text generation...")
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                    prompt = "The future of"
                    inputs = tokenizer(prompt, return_tensors="pt")

                    gen_output = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated_text = tokenizer.decode(
                        gen_output[0], skip_special_tokens=True
                    )
                    print(f"  ✓ Generation test: '{generated_text}'")

                except Exception as e:
                    print(f"  ❌ Inference test failed: {e}")
                    is_healthy = False

        # Final verdict
        print("\n" + "=" * 70)
        if is_healthy:
            print("✅ CHECKPOINT IS HEALTHY")
            print("=" * 70)
            print("\nThis checkpoint can be used for training.")
        else:
            print("❌ CHECKPOINT HAS DIVERGED")
            print("=" * 70)
            print("\nRecommendation: Reset this checkpoint using:")
            print(
                f"  python scripts/reset_diverged_model.py --checkpoint {checkpoint_path} --backup"
            )
        print("")

        return is_healthy

    except Exception as e:
        print(f"\n❌ Error checking checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Check checkpoint health")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint-fineweb-edu.pt",
        help="Path to checkpoint file",
    )

    args = parser.parse_args()

    is_healthy = check_checkpoint_health(args.checkpoint)

    return 0 if is_healthy else 1


if __name__ == "__main__":
    sys.exit(main())
