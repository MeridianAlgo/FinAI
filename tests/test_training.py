"""
Unit tests for training configuration and utilities
"""

import pytest
import yaml

from fin_ai.model import FinAIConfig
from fin_ai.training.trainer import DatasetCycler, TrainingConfig


@pytest.mark.unit
class TestTrainingConfig:
    """Test TrainingConfig class"""

    def test_default_config(self):
        """Test default training configuration"""
        config = TrainingConfig()

        assert config.batch_size == 4
        assert config.learning_rate == 3e-4
        assert config.max_steps == 100000
        assert config.warmup_steps == 1000

    def test_custom_config(self):
        """Test custom training configuration"""
        config = TrainingConfig(
            batch_size=8,
            learning_rate=1e-4,
            max_steps=1000,
        )

        assert config.batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.max_steps == 1000

    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file"""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "training": {
                "batch_size": 16,
                "learning_rate": 5e-4,
                "max_steps": 5000,
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = TrainingConfig.from_yaml(str(config_path))

        assert config.batch_size == 16
        assert config.learning_rate == 5e-4
        assert config.max_steps == 5000


@pytest.mark.unit
class TestModelConfig:
    """Test FinAIConfig class"""

    def test_model_config_from_yaml(self, tmp_path):
        """Test loading model config from YAML file"""
        config_path = tmp_path / "model_config.yaml"
        config_data = {
            "model": {
                "n_layers": 12,
                "embed_dim": 768,
                "n_heads": 12,
                "dropout": 0.2,
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = FinAIConfig.from_yaml(str(config_path))

        assert config.n_layers == 12
        assert config.embed_dim == 768
        assert config.n_heads == 12
        assert config.dropout == 0.2


@pytest.mark.unit
class TestDatasetCycler:
    """Test DatasetCycler class"""

    def test_dataset_cycler_initialization(self, tmp_path):
        """Test dataset cycler initialization"""
        # Create a test datasets config
        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {"name": "test_dataset_1", "split": "train"},
                {"name": "test_dataset_2", "split": "train"},
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        state_file = tmp_path / "dataset_state.json"
        cycler = DatasetCycler(str(config_path), state_file=str(state_file))

        assert cycler is not None
        assert len(cycler.datasets) == 2

    def test_dataset_cycler_current_dataset(self, tmp_path):
        """Test getting current dataset"""
        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {"name": "test_dataset_1", "split": "train"},
                {"name": "test_dataset_2", "split": "train"},
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        state_file = tmp_path / "dataset_state.json"
        cycler = DatasetCycler(str(config_path), state_file=str(state_file))

        current = cycler.get_current_dataset()
        assert current is not None
        assert "name" in current

    def test_dataset_offset_tracking(self, tmp_path):
        """Test dataset offset tracking"""
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

        # Test offset
        initial_offset = cycler.get_current_offset()
        assert initial_offset == 0

        # Increment offset
        cycler.increment_offset(100)
        assert cycler.get_current_offset() == 100
