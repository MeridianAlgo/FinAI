#!/usr/bin/env python3
"""
Fin.AI Text Generation Script

Usage:
    python generate.py --model checkpoints/model --prompt "Once upon a time"
"""

import argparse

import torch
from transformers import AutoTokenizer

from fin_ai.model import FinAIForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Generate text with Fin.AI")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model",
        help="Path to trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt to continue",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FinAIForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Encode prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated:\n{generated_text}")


if __name__ == "__main__":
    main()
