"""
Unit tests for FinAI model configuration
"""

import pytest

from fin_ai.model import FinAIConfig


@pytest.mark.unit
class TestFinAIConfig:
    """Test FinAIConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = FinAIConfig()

        assert config.vocab_size == 51200
        assert config.num_hidden_layers == 20
        assert config.hidden_size == 1280
        assert config.max_position_embeddings == 8192

    def test_custom_config(self):
        """Test custom configuration"""
        config = FinAIConfig(
            vocab_size=1000,
            num_hidden_layers=4,
            hidden_size=256,
        )

        assert config.vocab_size == 1000
        assert config.num_hidden_layers == 4
        assert config.hidden_size == 256

    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = FinAIConfig(vocab_size=1000, num_hidden_layers=4)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "vocab_size" in config_dict
        assert "num_hidden_layers" in config_dict
        assert config_dict["vocab_size"] == 1000
        assert config_dict["num_hidden_layers"] == 4

    def test_config_model_type(self):
        """Test model type is set correctly"""
        config = FinAIConfig()
        assert config.model_type == "finai"
