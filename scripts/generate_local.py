"""Local generation script for CPU-only environments without external tokenizers.

This maps characters to token ids via `ord(char) % vocab_size` and back via a printable ASCII map.
Run:
    python scripts/generate_local.py "Hello world" --max_new_tokens 20
"""

import argparse

import torch
from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel


def text_to_ids(text: str, vocab_size: int):
    return [ord(c) % vocab_size for c in text]


def ids_to_text(ids):
    chars = []
    for t in ids:
        c = t % 95 + 32
        chars.append(chr(c))
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Prompt text to generate from")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    cfg = FinAIConfig.from_preset(
        args.preset, vocab_size=args.vocab_size, max_seq_len=512
    )
    model = FinAIModel(cfg)
    model.eval()

    input_ids = torch.tensor(
        [text_to_ids(args.prompt, cfg.vocab_size)], dtype=torch.long
    )

    # allow repetition penalty and sampling controls
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
    )

    generated_ids = out[0].tolist()
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Generated (ids) ===")
    print(generated_ids)
    print("=== Generated (detokenized) ===")
    print(ids_to_text(generated_ids))


if __name__ == "__main__":
    main()
