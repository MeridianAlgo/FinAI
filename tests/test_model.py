"""Smoke tests for the tiny model factory used by the trainer tests.

Real training uses Qwen2.5-0.5B via AutoModelForCausalLM; this only checks that
build_smoke_model produces a working forward + generate path.
"""

import torch

from meridian.model import build_smoke_model


def test_forward_and_loss():
    model = build_smoke_model(vocab_size=500, hidden_size=64, num_layers=2)
    input_ids = torch.randint(0, 500, (2, 16))
    out = model(input_ids=input_ids, labels=input_ids.clone())
    assert out.loss is not None
    assert out.logits.shape == (2, 16, 500)


def test_generate():
    model = build_smoke_model(vocab_size=500, hidden_size=64, num_layers=2)
    input_ids = torch.randint(0, 500, (1, 8))
    out = model.generate(input_ids, max_new_tokens=5, do_sample=False)
    assert 8 <= out.shape[1] <= 13
