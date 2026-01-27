#!/usr/bin/env python3
"""
FinAI-Core v2.2 Text Generation Script
"""

import argparse
import torch
from transformers import AutoTokenizer
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.model.configuration_finai import FinAIConfig

def main():
    parser = argparse.ArgumentParser(description="Generate text with FinAI-Core v2.2")
    parser.add_argument("--model", type=str, default="checkpoints/model", help="Path to trained model")
    parser.add_argument("--prompt", type=str, default="The financial outlook for", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model}...")
    try:
        model = FinAIForCausalLM.from_pretrained(args.model)
    except Exception:
        print("Could not load from_pretrained, initializing from config for demo...")
        config = FinAIConfig()
        model = FinAIForCausalLM(config)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    print(f"\nPrompt: {args.prompt}\n" + "-"*50)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated:\n{generated_text}")

if __name__ == "__main__":
    main()
