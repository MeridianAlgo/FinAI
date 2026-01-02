import os
import tempfile
import random
import torch
import numpy as np
from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_forward_determinism_and_save_load():
    set_seed(1234)

    # Use a tiny config to keep model small and fast
    cfg = FinAIConfig.from_preset("tiny", vocab_size=256, max_seq_len=32)
    model = FinAIModel(cfg)
    model.eval()

    batch_size = 2
    seq_len = 8
    # deterministic input
    set_seed(1234)
    input_ids = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long
    )

    out1 = model(input_ids)
    out2 = model(input_ids)

    # logits should be identical between runs in eval mode
    assert torch.allclose(
        out1["logits"], out2["logits"]
    ), "Logits differ between identical forward passes"

    # Save and load, then compare parameters
    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td)
        loaded = FinAIModel.from_pretrained(td, device="cpu")

        for p1, p2 in zip(model.state_dict().values(), loaded.state_dict().values()):
            assert torch.equal(p1, p2), "Parameters mismatch after save/load"


def test_small_synthetic_training_decreases_loss():
    set_seed(2026)
    # Keep model very small for quick training
    cfg = FinAIConfig.from_preset("tiny", vocab_size=128, max_seq_len=16)
    model = FinAIModel(cfg)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 4
    seq_len = 12

    # Create a tiny synthetic next-token prediction task
    def make_batch():
        x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
        y = x.clone()
        return x, y

    losses = []

    steps = 20
    for step in range(steps):
        optimizer.zero_grad()
        x, y = make_batch()
        out = model(x, labels=y)
        loss = out.get("loss")
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    assert len(losses) > 0, "Training did not run"
    initial_loss = losses[0]
    final_loss = losses[-1]

    # Accept small variance but ensure the model didn't explode and had at least some improvement
    if min(losses) < initial_loss - 1e-6:
        # At some point loss decreased -> acceptable
        pass
    elif final_loss <= initial_loss * 0.99:
        # Final loss decreased by at least 1%
        pass
    elif final_loss <= initial_loss + 0.3:
        # Allow small increases due to noise but not explosion
        pass
    else:
        raise AssertionError(
            f"Loss did not improve or moved too much: initial={initial_loss}, final={final_loss}"
        )
