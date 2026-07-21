"""Tests for the training system."""

import pytest
import torch

from meridian.data.pipeline import create_smoke_dataloader
from meridian.model import build_smoke_model
from meridian.training.ewc import ElasticWeightConsolidation
from meridian.training.trainer import MeridianTrainer, TrainingConfig


@pytest.fixture
def small_setup():
    vocab_size = 500
    model = build_smoke_model(
        vocab_size=vocab_size, hidden_size=64, num_layers=2, max_position_embeddings=64
    )
    dataloader = create_smoke_dataloader(vocab_size, batch_size=2, block_size=32)
    return model, dataloader, vocab_size


class TestTrainer:
    def test_training_reduces_loss(self, small_setup, tmp_path):
        model, dataloader, _ = small_setup

        train_config = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=30,
            total_steps=30,
            learning_rate=5e-3,
            output_dir=str(tmp_path),
            save_steps=30,
            use_ewc=False,
            log_steps=30,
        )

        # Get initial loss
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            initial_loss = outputs.loss.item()

        # Train
        model.train()
        trainer = MeridianTrainer(model, dataloader, train_config)
        trainer.train()

        # Check loss decreased
        assert trainer.best_loss < initial_loss

    def test_checkpoint_save_load(self, small_setup, tmp_path):
        model, dataloader, _ = small_setup
        train_config = TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=5,
            total_steps=5,
            output_dir=str(tmp_path),
            save_steps=5,
            use_ewc=False,
        )

        trainer = MeridianTrainer(model, dataloader, train_config)
        trainer.global_step = 42
        trainer.save_checkpoint(str(tmp_path))

        trainer2 = MeridianTrainer(model, dataloader, train_config)
        success = trainer2.load_checkpoint(str(tmp_path))
        assert success
        assert trainer2.global_step == 42


class TestEWC:
    def test_ewc_penalty(self, small_setup):
        model, dataloader, _ = small_setup
        ewc = ElasticWeightConsolidation(model, ewc_lambda=10.0)
        ewc.compute_fisher(model, dataloader, max_samples=5)

        # Modify parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        penalty = ewc.penalty_value(model)
        assert penalty > 0

    def test_ewc_save_load(self, small_setup, tmp_path):
        model, dataloader, _ = small_setup
        ewc = ElasticWeightConsolidation(model)
        ewc.compute_fisher(model, dataloader, max_samples=5)

        path = str(tmp_path / "ewc.pt")
        ewc.save(path)

        ewc2 = ElasticWeightConsolidation(model)
        ewc2.load(path)
        assert ewc2._initialized
