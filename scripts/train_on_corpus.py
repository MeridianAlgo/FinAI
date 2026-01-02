"""Train the model on a tiny, structured corpus so it can learn readable language patterns.

This is CPU-friendly and intended to show qualitative improvement compared to random data.
"""

import os

import torch

from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel

CORPUS = [
    "Market volatility measures how much prices change over time.",
    "Value investing looks for undervalued companies with strong fundamentals.",
    "One way to hedge interest rate risk is to use duration-matching or interest rate swaps.",
    "Backtesting requires historical data, a clear strategy, and proper performance metrics.",
    "Common pitfalls when training on financial data are leakage and noisy labels.",
    "A good prompt for earnings-call summaries: 'Summarize the key points from the Q2 earnings call.'",
    "The Sharpe ratio measures risk-adjusted return.",
    "Overfitting occurs when a model learns noise instead of general patterns.",
    "Risk metrics include VaR, CVaR, and volatility.",
    "Preprocess text by cleaning, normalizing, and tokenizing consistently.",
]


def encode_sentence(tokenizer, sentence, max_len):
    # If tokenizer available, use it; otherwise simple char->id
    try:
        ids = tokenizer.encode(
            sentence, add_special_tokens=False, truncation=True, max_length=max_len
        )
    except Exception:
        ids = [ord(c) % 256 for c in sentence][:max_len]
    return ids


def train(steps=300, lr=5e-4, batch_size=4, seq_len=64):
    cfg = FinAIConfig.from_preset("tiny", vocab_size=50257, max_seq_len=seq_len)
    model = FinAIModel(cfg)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    # Try to use GPT-2 tokenizer if available for realistic ids
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        tokenizer = None

    dataset = []
    for s in CORPUS:
        ids = encode_sentence(tokenizer, s, max_len=seq_len)
        if len(ids) < 4:
            continue
        # pad/truncate

        ids = ids + [0] * (seq_len - len(ids)) if len(ids) < seq_len else ids[:seq_len]
        dataset.append(torch.tensor(ids, dtype=torch.long))

    assert dataset, "No data"

    def loader():
        i = 0
        while True:
            batch = []
            for _ in range(batch_size):
                batch.append(dataset[i % len(dataset)])
                i += 1
            yield torch.stack(batch)

    gen = loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        batch = next(gen)
        inputs = batch[:, :-1].to(device)
        labels = batch[:, 1:].to(device)

        optimizer.zero_grad()
        out = model(inputs, labels=labels)
        loss = out.get("loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    save_dir = os.path.join("checkpoints", "model")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"Saved trained-on-corpus model to {save_dir}")


if __name__ == "__main__":
    train(steps=300)
