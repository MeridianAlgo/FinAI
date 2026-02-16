"""Test generation from a trained MeridianFormer model."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from meridian.model.modeling import MeridianForCausalLM


def main():
    model_path = os.getenv("MODEL_PATH", "./checkpoint")

    print(f"Loading model from {model_path}...")
    model = MeridianForCausalLM.from_pretrained(model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    prompts = [
        "### Instruction:\nWhat is the current P/E ratio of Apple and what does it indicate?\n\n### Response:\n",
        "### Instruction:\nCalculate the compound interest on $10,000 at 5% over 10 years.\n\n### Response:\n",
        "### Instruction:\nExplain the Black-Scholes option pricing model.\n\n### Response:\n",
        "### Instruction:\nWhat are the key differences between stocks and bonds?\n\n### Response:\n",
    ]

    for prompt in prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt[:80]}...")
        tokens = tokenizer(prompt, return_tensors="pt")
        output = model.generate_text(tokens["input_ids"], max_new_tokens=128, temperature=0.7)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Output: {text}")


if __name__ == "__main__":
    main()
