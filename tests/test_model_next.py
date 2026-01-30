"""Basic smoke tests for FinAI-Next model"""
import torch
from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM


def test_model_initialization():
    """Test that model can be initialized"""
    config = FinAINextConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        tie_word_embeddings=False
    )
    model = FinAINextForCausalLM(config)
    assert model is not None
    assert model.config.tie_word_embeddings is False


def test_forward_pass():
    """Test that forward pass works"""
    config = FinAINextConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        tie_word_embeddings=False
    )
    model = FinAINextForCausalLM(config)
    input_ids = torch.randint(0, 1000, (2, 10))
    outputs = model(input_ids)
    assert outputs.logits is not None
    assert outputs.logits.shape == (2, 10, 1000)


def test_weights_not_tied():
    """Test that weights are not tied"""
    config = FinAINextConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        tie_word_embeddings=False
    )
    model = FinAINextForCausalLM(config)
    tied = model.model.embed_tokens.weight is model.lm_head.weight
    assert tied is False, "Weights should not be tied when tie_word_embeddings=False"


if __name__ == "__main__":
    test_model_initialization()
    test_forward_pass()
    test_weights_not_tied()
    print("All tests passed!")
