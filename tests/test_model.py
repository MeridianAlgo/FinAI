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
        assert len(model.layers) == sample_config.n_layers

    def test_model_forward(self, sample_model, device):
        """Test forward pass"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        with torch.no_grad():
            outputs = sample_model.model(input_ids=input_ids)

        assert outputs.last_hidden_state is not None
        assert outputs.last_hidden_state.shape == (
            batch_size,
            seq_len,
            sample_model.config.embed_dim,
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
        assert loaded_model.config.n_layers == sample_model.config.n_layers


@pytest.mark.unit
class TestModelComponents:
    """Test individual model components"""

    def test_attention_layer(self, sample_config, device):
        """Test attention mechanism"""
        from fin_ai.model.modeling_finai import FinAIAttention

        attention = FinAIAttention(sample_config).to(device)
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, sample_config.embed_dim, device=device
        )

        with torch.no_grad():
            output, _, _ = attention(hidden_states)

        assert output.shape == hidden_states.shape

    def test_mlp_layer(self, sample_config, device):
        """Test MLP/feedforward layer"""
        from fin_ai.model.modeling_finai import FinAIMLP

        mlp = FinAIMLP(sample_config).to(device)
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, sample_config.embed_dim, device=device
        )

        with torch.no_grad():
            output = mlp(hidden_states)

        assert output.shape == hidden_states.shape

    def test_rms_norm(self, sample_config, device):
        """Test RMSNorm layer"""
        from fin_ai.model.modeling_finai import FinAIRMSNorm

        norm = FinAIRMSNorm(sample_config.embed_dim).to(device)
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(
            batch_size, seq_len, sample_config.embed_dim, device=device
        )

        with torch.no_grad():
            output = norm(hidden_states)

        assert output.shape == hidden_states.shape

    def test_rotary_embeddings(self, sample_config, device):
        """Test rotary position embeddings"""
        from fin_ai.model.modeling_finai import FinAIRotaryEmbedding

        rope = FinAIRotaryEmbedding(
            dim=sample_config.embed_dim // sample_config.n_heads,
            max_position_embeddings=sample_config.max_seq_len,
        ).to(device)

        seq_len = 16
        hidden_states = torch.randn(2, seq_len, sample_config.embed_dim, device=device)

        cos, sin = rope(hidden_states, seq_len=seq_len)

        assert cos.shape[0] == seq_len
        assert sin.shape[0] == seq_len
