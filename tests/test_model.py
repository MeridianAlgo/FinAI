"""
Unit tests for Fin.AI model architecture
"""

import pytest
import torch

from fin_ai.model import FinAIForCausalLM, FinAIModel


@pytest.mark.unit
class TestFinAIModel:
    """Test FinAIModel class"""

    def test_model_initialization(self, sample_config):
        """Test model can be initialized"""
        model = FinAIModel(sample_config)
        assert model is not None
        assert len(model.layers) == sample_config.num_hidden_layers

    def test_model_forward(self, sample_model, device):
        """Test forward pass"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = sample_model.model(input_ids=input_ids)

        assert output is not None
        assert output.shape == (
            batch_size,
            seq_len,
            sample_model.config.hidden_size,
        )

    def test_model_embeddings(self, sample_model):
        """Test embedding layer"""
        embeddings = sample_model.get_input_embeddings()
        assert embeddings is not None
        assert embeddings.num_embeddings == sample_model.config.vocab_size


@pytest.mark.unit
class TestFinAIForCausalLM:
    """Test FinAIForCausalLM class"""

    def test_causal_lm_initialization(self, sample_config):
        """Test causal LM model initialization"""
        model = FinAIForCausalLM(sample_config)
        assert model is not None
        assert model.lm_head is not None

    def test_causal_lm_forward(self, sample_model, sample_batch):
        """Test forward pass with labels"""
        with torch.no_grad():
            outputs = sample_model(**sample_batch)

        assert outputs.loss is not None
        assert outputs.logits is not None
        assert outputs.logits.shape[-1] == sample_model.config.vocab_size

    def test_causal_lm_forward_inference(self, sample_model, device):
        """Test forward pass in inference mode"""
        sample_model.eval()
        input_ids = torch.randint(0, 1000, (1, 10), device=device)

        with torch.no_grad():
            outputs = sample_model(input_ids=input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[1] == 10
        assert outputs.logits.shape[2] == sample_model.config.vocab_size

    def test_model_save_load(self, sample_model, tmp_path):
        """Test model saving and loading"""
        save_path = tmp_path / "test_model"

        # Save model
        sample_model.save_pretrained(save_path)

        # Load model
        loaded_model = FinAIForCausalLM.from_pretrained(save_path)

        assert loaded_model is not None
        assert loaded_model.config.vocab_size == sample_model.config.vocab_size
        assert loaded_model.config.num_hidden_layers == sample_model.config.num_hidden_layers


@pytest.mark.unit
class TestModelComponents:
    """Test individual model components"""

    def test_mla_attention(self, sample_config, device):
        """Test MLA attention mechanism"""
        from fin_ai.model.modeling_finai import MLAAttention

        attention = MLAAttention(sample_config).to(device)
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, sample_config.hidden_size, device=device
        )

        with torch.no_grad():
            output = attention(hidden_states)

        assert output.shape == hidden_states.shape

    def test_mamba_block(self, sample_config, device):
        """Test Mamba block"""
        from fin_ai.model.modeling_finai import Mamba2Block

        mamba = Mamba2Block(sample_config).to(device)
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, sample_config.hidden_size, device=device
        )

        with torch.no_grad():
            output = mamba(hidden_states)

        assert output.shape == hidden_states.shape

    def test_moe_layer(self, sample_config, device):
        """Test MoE layer"""
        from fin_ai.model.modeling_finai import DeepSeekMoE

        moe = DeepSeekMoE(sample_config).to(device)
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, sample_config.hidden_size, device=device
        )

        with torch.no_grad():
            output = moe(hidden_states)

        assert output.shape == hidden_states.shape

    def test_rms_norm(self, sample_config, device):
        """Test RMSNorm layer"""
        from fin_ai.model.modeling_finai import FinAIRMSNorm

        norm = FinAIRMSNorm(sample_config.hidden_size).to(device)
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, sample_config.hidden_size, device=device
        )

        with torch.no_grad():
            output = norm(hidden_states)

        assert output.shape == hidden_states.shape
