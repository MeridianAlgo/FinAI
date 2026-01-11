"""
Unit tests for Fin.AI model configuration
"""

import pytest

from fin_ai.model import FinAIConfig


@pytest.mark.unit
class TestFinAIConfig:
    """Test FinAIConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = FinAIConfig()

        assert config.vocab_size == 50257
        assert config.n_layers == 8
        assert config.n_heads == 8
        assert config.n_kv_heads == 4
        assert config.embed_dim == 512
        assert config.max_seq_len == 2048
        assert config.use_flash_attention is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = FinAIConfig(
            vocab_size=1000,
            n_layers=4,
            n_heads=4,
            embed_dim=256,
        )

        assert config.vocab_size == 1000
        assert config.n_layers == 4
        assert config.n_heads == 4
        assert config.embed_dim == 256

    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = FinAIConfig(vocab_size=1000, n_layers=4)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "vocab_size" in config_dict
        assert "n_layers" in config_dict
        assert config_dict["vocab_size"] == 1000
        assert config_dict["n_layers"] == 4

    def test_config_model_type(self):
        """Test model type is set correctly"""
        config = FinAIConfig()
        assert config.model_type == "finai"
