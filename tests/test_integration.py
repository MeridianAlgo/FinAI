"""
Integration tests for Fin.AI
"""

import pytest
import torch

from fin_ai.model import FinAIConfig, FinAIForCausalLM


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests"""

    def test_training_step(self, sample_model, sample_batch, device):
        """Test a single training step"""
        sample_model.train()
        optimizer = torch.optim.AdamW(sample_model.parameters(), lr=1e-4)

        # Forward pass
        outputs = sample_model(**sample_batch)
        loss = outputs.loss

        assert loss is not None
        assert loss.item() > 0

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_inference_pipeline(self, device):
        """Test complete inference pipeline"""
        # Create a small model
        config = FinAIConfig(
            vocab_size=1000,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            embed_dim=128,
            max_seq_len=64,
        )
        model = FinAIForCausalLM(config).to(device)
        model.eval()

        # Run inference
        input_ids = torch.randint(0, 1000, (1, 10), device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1
        assert outputs.logits.shape[1] == 10

    @pytest.mark.slow
    def test_save_and_load_workflow(self, tmp_path, device):
        """Test complete save and load workflow"""
        # Create and train a model
        config = FinAIConfig(
            vocab_size=1000,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            embed_dim=128,
        )
        model = FinAIForCausalLM(config).to(device)

        # Save model
        save_path = tmp_path / "model"
        model.save_pretrained(save_path)
        config.save_pretrained(save_path)

        # Load model
        loaded_model = FinAIForCausalLM.from_pretrained(save_path).to(device)
        loaded_config = FinAIConfig.from_pretrained(save_path)

        # Verify
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.n_layers == config.n_layers

        # Test inference with loaded model
        input_ids = torch.randint(0, 1000, (1, 10), device=device)
        with torch.no_grad():
            outputs = loaded_model(input_ids=input_ids)

        assert outputs.logits is not None
        assert outputs.logits.shape[1] == 10
