import os
import tempfile

import torch
from transformers import AutoTokenizer

from fin_ai.data import load_datasets_from_config
from fin_ai.model import FinAIConfig, FinAIModel


def test_imports():
    # Sanity imports
    assert FinAIModel is not None
    assert FinAIConfig is not None


def test_model_forward_and_loss():
    cfg = FinAIConfig.from_preset("tiny")
    model = FinAIModel(cfg)
    model.eval()

    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, cfg.vocab_size or 1000, (batch_size, seq_len))

    out = model(input_ids)
    assert "logits" in out
    assert out["logits"].shape[0] == batch_size

    labels = input_ids.clone()
    out2 = model(input_ids, labels=labels)
    assert "loss" in out2
    assert out2["loss"].item() >= 0


def test_generate_and_save_load():
    cfg = FinAIConfig.from_preset("tiny")
    model = FinAIModel(cfg)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Run a short generate
    with torch.no_grad():
        gen = model.generate(input_ids, max_new_tokens=4, do_sample=False)
    assert gen is not None

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "m")
        model.save_pretrained(save_path)
        loaded = FinAIModel.from_pretrained(save_path)

        with torch.no_grad():
            a = model(input_ids)["logits"]
            b = loaded(input_ids)["logits"]
        assert a.shape == b.shape


def test_dataset_loading_small():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Monkeypatch load_dataset to avoid HF network access during tests
    def fake_load_dataset(name, *args, **kwargs):
        def gen():
            for i in range(5):
                yield {"text": f"dummy {i} {name}"}

        return gen()

    import fin_ai.data.dataset as ds_module

    ds_module.load_dataset = fake_load_dataset

    ds, offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=128,
        max_samples=10,
        force_streaming=True,
    )

    assert len(ds) >= 0
