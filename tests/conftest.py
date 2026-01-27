"""
Shared test fixtures and configuration for Fin.AI tests
"""

import pytest
import torch


@pytest.fixture
def device():
    """Get the appropriate device for testing"""
    return torch.device("cpu")  # Use CPU for tests


@pytest.fixture
def sample_config():
    """Sample model configuration for testing"""
    from fin_ai.model import FinAIConfig

    return FinAIConfig(
        vocab_size=1000,  # Small vocab for testing
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        embed_dim=128,
        ff_dim=512,
        max_seq_len=128,
        dropout=0.1,
        tie_word_embeddings=False,  # Disable for easier testing with safetensors
    )


@pytest.fixture
def sample_model(sample_config, device):
    """Create a small model for testing"""
    from fin_ai.model import FinAIForCausalLM

    model = FinAIForCausalLM(sample_config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing"""
    batch_size = 2
    seq_len = 32
    vocab_size = 1000

    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
    }
