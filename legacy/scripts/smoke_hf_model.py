import os

import torch
from transformers import AutoTokenizer

from fin_ai.model import FinAIForCausalLM


def main():
    model_dir = os.path.join("checkpoints", "model")
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise RuntimeError(f"Missing {os.path.join(model_dir, 'config.json')}")

    model = FinAIForCausalLM.from_pretrained(model_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = "The market"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        _ = model(**inputs, labels=inputs["input_ids"])
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)

    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
