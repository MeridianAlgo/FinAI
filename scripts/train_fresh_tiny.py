"""Train a fresh tiny model for a short run (CPU-friendly) and save to `checkpoints/model`.

This is a smoke/validation training run with slightly improved defaults.
"""

import os

import torch
from fin_ai.model.transformer import FinAIModel

from fin_ai.model.config import FinAIConfig


def synthetic_dataloader(vocab_size, seq_len, dataset_size=1000, batch_size=8):
    # Create synthetic dataset of simple token sequences
    import random

    data = []
    for _ in range(dataset_size):
        seq = torch.tensor(
            [random.randint(0, vocab_size - 1) for _ in range(seq_len)],
            dtype=torch.long,
        )
        data.append(seq)

    def iterator():
        i = 0
        while True:
            batch = []
            for _ in range(batch_size):
                batch.append(data[i % len(data)])
                i += 1
            batch = torch.stack(batch)
            yield batch

    return iterator()


def train(steps=200, lr=5e-4, batch_size=8, seq_len=32):
    cfg = FinAIConfig.from_preset("tiny", vocab_size=256, max_seq_len=seq_len)
    model = FinAIModel(cfg)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loader = synthetic_dataloader(
        cfg.vocab_size, seq_len, dataset_size=500, batch_size=batch_size
    )

    for step in range(steps):
        batch = next(loader)
        inputs = batch[:, :-1].to(device)
        labels = batch[:, 1:].to(device)

        optimizer.zero_grad()
        out = model(inputs, labels=labels)
        loss = out.get("loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    # Save to HF-style directory
    save_dir = os.path.join("checkpoints", "model")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"Saved fresh model to {save_dir}")


if __name__ == "__main__":
    train(steps=200, lr=5e-4, batch_size=6, seq_len=64)
