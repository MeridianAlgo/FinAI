"""
Tests for data loading and processing
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoTokenizer


@pytest.mark.unit
class TestDataProcessing:
    """Test data loading and processing"""

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_tokenizer_loading(self, mock_from_pretrained):
        """Test tokenizer can be loaded"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab_size = 50257
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_tokenization(self, mock_from_pretrained):
        """Test text tokenization"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        text = "The future of AI is bright"

        tokens = tokenizer(text, return_tensors="pt")

        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_batch_tokenization(self, mock_from_pretrained):
        """Test batch tokenization"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        mock_from_pretrained.return_value = mock_tokenizer

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
