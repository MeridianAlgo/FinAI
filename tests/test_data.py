"""
Tests for data loading and processing
"""

import pytest
from transformers import AutoTokenizer


@pytest.mark.unit
class TestDataProcessing:
    """Test data loading and processing"""

    def test_tokenizer_loading(self):
        """Test tokenizer can be loaded"""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    def test_tokenization(self):
        """Test text tokenization"""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        text = "The future of AI is bright"

        tokens = tokenizer(text, return_tensors="pt")

        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1

    def test_batch_tokenization(self):
        """Test batch tokenization"""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        texts = [
            "The future of AI is bright",
            "Machine learning is transforming the world",
        ]

        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        assert tokens["input_ids"].shape[0] == 2
        assert tokens["attention_mask"].shape[0] == 2
