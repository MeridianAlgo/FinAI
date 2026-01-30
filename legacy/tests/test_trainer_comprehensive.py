"""
Comprehensive tests for training module
"""

import json
import os

import pytest
import torch
import yaml

from fin_ai.model import FinAIConfig, FinAIForCausalLM
from fin_ai.training.trainer import DatasetCycler, FinAITrainer, TrainingConfig


@pytest.mark.unit
class TestTrainingConfigComprehensive:
    """Comprehensive tests for TrainingConfig"""

    def test_training_config_attributes(self):
        """Test all training config attributes"""
        config = TrainingConfig(
            batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=500,
            max_steps=5000,
            max_grad_norm=1.5,
            eval_steps=100,
            save_steps=500,
        )

        assert config.batch_size == 8
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.warmup_steps == 500
        assert config.max_steps == 5000
        assert config.max_grad_norm == 1.5
        assert config.eval_steps == 100
        assert config.save_steps == 500

    def test_training_config_optimizer_settings(self):
        """Test optimizer-specific settings"""
        config = TrainingConfig(
            optimizer="adamw",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )

        assert config.optimizer == "adamw"
        assert config.adam_beta1 == 0.9
        assert config.adam_beta2 == 0.999
        assert config.adam_epsilon == 1e-8

    def test_training_config_from_yaml_complete(self, tmp_path):
        """Test loading complete training config from YAML"""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "training": {
                "batch_size": 16,
                "learning_rate": 5e-4,
                "max_steps": 10000,
                "warmup_steps": 2000,
                "eval_steps": 500,
                "save_steps": 1000,
            },
            "checkpointing": {
                "save_total_limit": 5,
                "resume_from_checkpoint": True,
            },
            "logging": {
                "log_steps": 50,
                "use_comet": True,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = TrainingConfig.from_yaml(str(config_path))

        assert config.batch_size == 16
        assert config.learning_rate == 5e-4
        assert config.max_steps == 10000
        assert config.warmup_steps == 2000
        assert config.eval_steps == 500
        assert config.save_steps == 1000
        assert config.save_total_limit == 5
        assert config.resume_from_checkpoint is True
        assert config.log_steps == 50
        assert config.use_comet is True


@pytest.mark.unit
class TestDatasetCyclerComprehensive:
    """Comprehensive tests for DatasetCycler"""

    def test_dataset_cycler_multiple_datasets(self, tmp_path):
        """Test cycler with multiple datasets"""
        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {"name": "dataset_1", "split": "train"},
                {"name": "dataset_2", "split": "train"},
                {"name": "dataset_3", "split": "train"},
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        state_file = tmp_path / "dataset_state.json"
        cycler = DatasetCycler(str(config_path), state_file=str(state_file))

        assert len(cycler.datasets) == 3
        assert cycler.current_dataset_name is not None

    def test_dataset_cycler_offset_persistence(self, tmp_path):
        """Test that offsets persist across instances"""
        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {"name": "dataset_1", "split": "train"},
                {"name": "dataset_2", "split": "train"},
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        state_file = tmp_path / "dataset_state.json"

        # First instance
        cycler1 = DatasetCycler(str(config_path), state_file=str(state_file))
        cycler1.increment_offset(100)
        offset1 = cycler1.get_current_offset()

        # Second instance should load the same offset
        cycler2 = DatasetCycler(str(config_path), state_file=str(state_file))
        offset2 = cycler2.get_current_offset()

        assert offset1 == offset2 == 100

    def test_dataset_cycler_increment_offset(self, tmp_path):
        """Test incrementing dataset offset"""
        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {"name": "test_dataset", "split": "train"},
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        state_file = tmp_path / "dataset_state.json"
        cycler = DatasetCycler(str(config_path), state_file=str(state_file))

        assert cycler.get_current_offset() == 0

        cycler.increment_offset(50)
        assert cycler.get_current_offset() == 50

        cycler.increment_offset(50)
        assert cycler.get_current_offset() == 100

    def test_dataset_cycler_state_file_creation(self, tmp_path):
        """Test that state file is created"""
        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {"name": "test_dataset", "split": "train"},
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        state_file = tmp_path / "dataset_state.json"
        cycler = DatasetCycler(str(config_path), state_file=str(state_file))

        assert os.path.exists(state_file)

        with open(state_file, "r") as f:
            state = json.load(f)

        assert "current_dataset_idx" in state
        assert "dataset_offsets" in state

    def test_dataset_cycler_get_current_dataset(self, tmp_path):
        """Test getting current dataset info"""
        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {"name": "dataset_1", "split": "train"},
                {"name": "dataset_2", "split": "train"},
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        state_file = tmp_path / "dataset_state.json"
        cycler = DatasetCycler(str(config_path), state_file=str(state_file))

        current = cycler.get_current_dataset()
        assert current is not None
        assert "name" in current
        assert "split" in current


@pytest.mark.unit
class TestFinAITrainerInitialization:
    """Test FinAITrainer initialization and setup"""

    def test_trainer_initialization(self, sample_model, sample_batch):
        """Test trainer can be initialized"""
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
            sample_batch["labels"],
        )
        dataloader = DataLoader(dataset, batch_size=2)

        config = TrainingConfig(max_steps=100, batch_size=2)

        trainer = FinAITrainer(
            model=sample_model,
            train_dataloader=dataloader,
            config=config,
        )

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_trainer_device_placement(self, sample_model):
        """Test trainer places model on correct device"""
        from torch.utils.data import DataLoader, TensorDataset

        batch_size = 2
        seq_len = 32
        vocab_size = 1000

        dataset = TensorDataset(
            torch.randint(0, vocab_size, (batch_size, seq_len)),
            torch.ones(batch_size, seq_len),
            torch.randint(0, vocab_size, (batch_size, seq_len)),
        )
        dataloader = DataLoader(dataset, batch_size=2)

        config = TrainingConfig(max_steps=10, batch_size=2)

        trainer = FinAITrainer(
            model=sample_model,
            train_dataloader=dataloader,
            config=config,
        )

        assert trainer.device.type == "cpu"

    def test_trainer_optimizer_creation(self, sample_model):
        """Test optimizer is created correctly"""
        from torch.utils.data import DataLoader, TensorDataset

        batch_size = 2
        seq_len = 32
        vocab_size = 1000

        dataset = TensorDataset(
            torch.randint(0, vocab_size, (batch_size, seq_len)),
            torch.ones(batch_size, seq_len),
            torch.randint(0, vocab_size, (batch_size, seq_len)),
        )
        dataloader = DataLoader(dataset, batch_size=2)

        trainer = FinAITrainer(
            model=sample_model,
            train_dataloader=dataloader,
            config=TrainingConfig(max_steps=10),
        )

        assert trainer.optimizer is not None
        # Support both standard AdamW and bitsandbytes 8-bit AdamW
        opt_name = type(trainer.optimizer).__name__
        assert "AdamW" in opt_name

    def test_trainer_scheduler_creation(self, sample_model):
        """Test trainer creates learning rate scheduler"""
        from torch.utils.data import DataLoader, TensorDataset

        batch_size = 2
        seq_len = 32
        vocab_size = 1000

        dataset = TensorDataset(
            torch.randint(0, vocab_size, (batch_size, seq_len)),
            torch.ones(batch_size, seq_len),
            torch.randint(0, vocab_size, (batch_size, seq_len)),
        )
        dataloader = DataLoader(dataset, batch_size=2)

        config = TrainingConfig(
            max_steps=100,
            batch_size=2,
            warmup_steps=10,
        )

        trainer = FinAITrainer(
            model=sample_model,
            train_dataloader=dataloader,
            config=config,
        )

        assert trainer.scheduler is not None


@pytest.mark.unit
class TestTrainerUtilities:
    """Test trainer utility functions"""

    def test_get_hf_token_from_env(self, monkeypatch):
        """Test getting HF token from environment"""
        from torch.utils.data import DataLoader, TensorDataset

        monkeypatch.setenv("HF_TOKEN", "test_token_123")

        batch_size = 2
        seq_len = 32
        vocab_size = 1000

        dataset = TensorDataset(
            torch.randint(0, vocab_size, (batch_size, seq_len)),
            torch.ones(batch_size, seq_len),
            torch.randint(0, vocab_size, (batch_size, seq_len)),
        )
        dataloader = DataLoader(dataset, batch_size=2)

        config = FinAIConfig(vocab_size=1000, n_layers=2, n_heads=4, n_kv_heads=2)
        model = FinAIForCausalLM(config)

        trainer = FinAITrainer(
            model=model,
            train_dataloader=dataloader,
            config=TrainingConfig(max_steps=10),
        )

        token = trainer._get_hf_token()
        assert token == "test_token_123"

    def test_get_hf_token_from_env_file(self, tmp_path, monkeypatch):
        """Test getting HF token from .env file"""
        from torch.utils.data import DataLoader, TensorDataset

        # Ensure HF_TOKEN is not in env so we test the file reading
        monkeypatch.delenv("HF_TOKEN", raising=False)

        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("HF_TOKEN=token_from_env_file\n")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        batch_size = 2
        seq_len = 32
        vocab_size = 1000

        dataset = TensorDataset(
            torch.randint(0, vocab_size, (batch_size, seq_len)),
            torch.ones(batch_size, seq_len),
            torch.randint(0, vocab_size, (batch_size, seq_len)),
        )
        dataloader = DataLoader(dataset, batch_size=2)

        config = FinAIConfig(vocab_size=1000, n_layers=2, n_heads=4, n_kv_heads=2)
        model = FinAIForCausalLM(config)

        trainer = FinAITrainer(
            model=model,
            train_dataloader=dataloader,
            config=TrainingConfig(max_steps=10),
        )

        token = trainer._get_hf_token()
        assert token == "token_from_env_file"
