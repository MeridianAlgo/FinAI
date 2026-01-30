"""
Shared test fixtures and configuration for FinAI tests
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
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=128,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        tie_word_embeddings=False,
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
