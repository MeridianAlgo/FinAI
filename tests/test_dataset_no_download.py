import os
import tempfile
from pathlib import Path
import pytest

from fin_ai.data.dataset import load_datasets_from_config
from transformers import AutoTokenizer


def test_force_streaming_does_not_write_cache(monkeypatch, tmp_path):
    """Ensure streaming mode does not write to HF datasets cache on disk."""
    # Make a temporary cache dir and ensure it's empty
    cache_dir = tmp_path / "hf_cache"
    cache_dir.mkdir()

    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_dir))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Monkeypatch `datasets.load_dataset` to avoid network access entirely.
    def fake_load_dataset(name, *args, **kwargs):
        # Return a small iterable that mimics streaming dataset items
        def gen():
            for i in range(20):
                yield {"text": f"Sample text {i} for {name}"}

        return gen()

    monkeypatch.setattr(
        "fin_ai.data.dataset.load_dataset",
        lambda *a, **k: fake_load_dataset(a[0], *a[1:], **k),
    )

    # Call loader with force_streaming True (should not write to HF cache)
    ds, offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=64,
        max_samples=2,
        offset=0,
        force_streaming=True,
    )

    # Since we monkeypatched, no network downloads occurred and cache_dir should be empty
    large_files = [
        p for p in cache_dir.rglob("*") if p.is_file() and p.stat().st_size > 1_000_000
    ]
    assert (
        len(large_files) == 0
    ), f"Found large cache files when streaming: {large_files}"
