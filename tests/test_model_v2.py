"""Tests for Fin.AI v2 model architecture"""

import pytest
import torch
from fin_ai.model import FinAIModel, FinAIConfig


def test_model_creation():
    """Test that model can be created"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    assert model is not None
    assert isinstance(model, FinAIModel)


def test_model_forward():
    """Test forward pass"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)


def test_model_forward_with_labels():
    """Test forward pass with loss computation"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["loss"].item() > 0


def test_model_generation():
    """Test text generation"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
    )
    
    assert generated.shape[0] == 1
    assert generated.shape[1] == 30  # 10 + 20


def test_model_save_load(tmp_path):
    """Test saving and loading model"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    # Save
    save_path = tmp_path / "model"
    model.save_pretrained(str(save_path))
    
    # Load
    loaded_model = FinAIModel.from_pretrained(str(save_path))
    
    # Test that loaded model works
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    outputs = loaded_model(input_ids)
    assert "logits" in outputs


def test_grouped_query_attention():
    """Test that GQA works correctly"""
    config = FinAIConfig(
        vocab_size=1000,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,  # Half the Q heads
        embed_dim=256,
        ff_dim=896,
        max_seq_len=128,
    )
    model = FinAIModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    outputs = model(input_ids)
    
    assert outputs["logits"].shape == (2, 32, config.vocab_size)


def test_parameter_count():
    """Test parameter counting"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    actual_params = model.count_parameters()
    estimated_params = config.num_parameters
    
    # Should be within 5% of estimate
    assert abs(actual_params - estimated_params) / estimated_params < 0.05


def test_all_presets():
    """Test all model presets"""
    presets = ["tiny", "small", "medium", "large"]
    
    for preset in presets:
        config = FinAIConfig.from_preset(preset)
        model = FinAIModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        outputs = model(input_ids)
        
        assert "logits" in outputs
        print(f"{preset}: {model.count_parameters():,} parameters")


def test_attention_mask():
    """Test that attention mask works"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 20:] = 0  # Mask second half of first sequence
    
    outputs = model(input_ids, attention_mask=attention_mask)
    
    assert "logits" in outputs


def test_gradient_flow():
    """Test that gradients flow properly"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = input_ids.clone()
    
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    
    loss.backward()
    
    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_repetition_penalty():
    """Test generation with repetition penalty"""
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Generate with high repetition penalty
    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=1.0,
        repetition_penalty=2.0,
    )
    
    assert generated.shape[1] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
