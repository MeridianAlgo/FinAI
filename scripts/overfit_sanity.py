"""Tiny overfit script: trains the FinAIModel on a tiny synthetic dataset to ensure
training loop, loss, and label alignment are correct.

Run locally (quick):
    python scripts/overfit_sanity.py
"""
import tempfile
import torch
from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel


def overfit_steps(steps=50, lr=1e-3, batch_size=8, seq_len=16):
    cfg = FinAIConfig.from_preset("tiny", vocab_size=256, max_seq_len=seq_len)
    model = FinAIModel(cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # tiny synthetic dataset: a handful of sequences to memorize
    data = [torch.randint(0, cfg.vocab_size, (seq_len,), dtype=torch.long) for _ in range(16)]

    losses = []
    for step in range(steps):
        batch_idxs = torch.randint(0, len(data), (batch_size,))
        batch = torch.stack([data[int(i)] for i in batch_idxs])
        inputs = batch[:, :-1]
        labels = batch[:, 1:]

        optimizer.zero_grad()
        out = model(inputs, labels=labels)
        loss = out.get("loss")
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 10 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    return model, losses


def main():
    print("Running tiny overfit sanity test (CPU). This should quickly decrease loss.")
    model, losses = overfit_steps(steps=60)
    print("Done. Loss sample:", losses[:5], "...", losses[-5:])

    # Save a tiny checkpoint so we can inspect it
    chk = {
        "model_state_dict": model.state_dict(),
    }
    out_dir = "checkpoints"
    import os

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sanity-overfit.pt")
    torch.save(chk, path)
    print(f"Saved tiny checkpoint to {path}")


if __name__ == "__main__":
    main()
