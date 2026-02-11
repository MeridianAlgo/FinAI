"""Tests for MeridianFormer model architecture."""

import torch
import pytest

from meridian.model.configuration import MeridianConfig
from meridian.model.modeling import (
    MeridianForCausalLM,
    RMSNorm,
    MeridianSwiGLU,
    MeridianMoELayer,
    NumeracyEncoder,
)


@pytest.fixture
def small_config():
    """Create a small config for fast testing."""
    return MeridianConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=176,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_token=2,
        expert_intermediate_size=88,
        moe_layer_frequency=2,
        max_position_embeddings=128,
        gradient_checkpointing=False,
        use_numeracy_encoding=True,
        numeracy_embed_dim=16,
        tie_word_embeddings=False,
    )


@pytest.fixture
def small_model(small_config):
    return MeridianForCausalLM(small_config)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64) * 100
        out = norm(x)
        # Output should be roughly normalized
        assert out.abs().mean() < 10


class TestSwiGLU:
    def test_output_shape(self):
        ffn = MeridianSwiGLU(64, 176)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)


class TestMoE:
    def test_output_shape(self, small_config):
        moe = MeridianMoELayer(small_config)
        x = torch.randn(2, 10, 64)
        out, aux_loss = moe(x)
        assert out.shape == (2, 10, 64)
        assert aux_loss.ndim == 0  # Scalar

    def test_aux_loss_positive(self, small_config):
        moe = MeridianMoELayer(small_config)
        x = torch.randn(2, 10, 64)
        _, aux_loss = moe(x)
        assert aux_loss >= 0


class TestNumeracy:
    def test_output_shape(self):
        enc = NumeracyEncoder(64, 16, 1000)
        x = torch.randn(2, 10, 64)
        ids = torch.randint(0, 1000, (2, 10))
        out = enc(x, ids)
        assert out.shape == (2, 10, 64)


class TestModel:
    def test_forward(self, small_model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        labels = input_ids.clone()
        outputs = small_model(input_ids=input_ids, labels=labels)
        assert outputs.loss is not None
        assert outputs.logits.shape == (2, 16, small_config.vocab_size)

    def test_generate(self, small_model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (1, 8))
        output = small_model.generate_text(input_ids, max_new_tokens=5, temperature=1.0)
        assert output.shape[1] >= 8  # At least input length
        assert output.shape[1] <= 13  # At most input + max_new

    def test_gradient_flow(self, small_model, small_config):
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        labels = input_ids.clone()
        outputs = small_model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()

        # Check gradients exist
        has_grad = False
        for param in small_model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_save_load(self, small_model, tmp_path):
        save_path = str(tmp_path / "test_model")
        small_model.save_pretrained(save_path, safe_serialization=True)

        loaded = MeridianForCausalLM.from_pretrained(save_path)

        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            small_model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_param_count(self, small_model):
        total = sum(p.numel() for p in small_model.parameters())
        assert total > 0
        print(f"  Small model params: {total:,}")
