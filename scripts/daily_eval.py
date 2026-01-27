#!/usr/bin/env python3
"""
Daily Model Evaluation Script

Tests the model with a consistent prompt to track evolution over time.
Can be run locally or in CI.

Usage:
    python scripts/daily_eval.py --model-path checkpoints/model
    python scripts/daily_eval.py --hf-repo MeridianAlgo/FinAI-Lite
"""

import argparse
import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_model(model_path: str, prompt: str, max_new_tokens: int = 100):
    """Evaluate model with given prompt."""
    print(f"Loading model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    print(f"\n{'=' * 60}")
    print(f"DAILY EVALUATION - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'=' * 60}")
    print(f"Prompt: {prompt}")
    print(f"{'=' * 60}\n")

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated Text:\n{generated_text}\n")
    print(f"{'=' * 60}\n")

    return generated_text


def save_result(
    prompt: str,
    response: str,
    model_params: str,
    history_file: str = "daily_eval_history.json",
):
    """Save evaluation result to history file."""
    result = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "response": response,
        "model_params": model_params,
    }

    # Load existing history
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

    history.append(result)

    # Keep only last 30 days
    history = history[-30:]

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Result saved to {history_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Daily model evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/model",
        help="Path to model directory",
    )
    parser.add_argument(
        "--hf-repo", type=str, help="Hugging Face repo ID (alternative to --model-path)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Test prompt",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=100, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--history-file",
        type=str,
        default="daily_eval_history.json",
        help="Path to history file",
    )
    args = parser.parse_args()

    # Download from HF if specified
    if args.hf_repo:
        from huggingface_hub import snapshot_download

        print(f"Downloading model from {args.hf_repo}...")
        model_path = snapshot_download(
            repo_id=args.hf_repo,
            local_dir="eval_model_temp",
            ignore_patterns=["*.md", "*.txt"],
        )
        args.model_path = model_path

    # Evaluate
    try:
        response = evaluate_model(args.model_path, args.prompt, args.max_new_tokens)

        # Get model params if available
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            model_params = (
                str(config.num_parameters)
                if hasattr(config, "num_parameters")
                else "unknown"
            )
        except:
            model_params = "unknown"

        # Save result
        save_result(args.prompt, response, model_params, args.history_file)

        print("Evaluation complete!")

    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
