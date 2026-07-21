"""Test generation from a trained Meridian.AI model."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-0.5B")
    tokenizer_id = os.getenv("TOKENIZER_ID", model_path)

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
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
        input_len = int(tokens["input_ids"].shape[-1])
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
                max_new_tokens=128,
                min_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_len = int(output_ids.shape[-1])
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        text_with_special = tokenizer.decode(output_ids[0], skip_special_tokens=False)

        continuation = text[len(prompt) :] if text.startswith(prompt) else text

        print(f"Output: {continuation.strip()}")
        print(
            f"[DEBUG] input_len={input_len} output_len={output_len} new_tokens={output_len - input_len}"
        )
        print(f"[DEBUG] decoded_with_special_repr={text_with_special!r}")
        print(f"[DEBUG] continuation_repr={continuation!r}")


if __name__ == "__main__":
    main()
