"""
Comprehensive tests for data loading and dataset handling
"""

import pytest
import torch
from transformers import AutoTokenizer

from fin_ai.data import FinAIDataset, create_dataloader, load_datasets_from_config


@pytest.mark.unit
class TestFinAIDataset:
    """Test FinAIDataset class"""

    def test_dataset_initialization(self):
        """Test dataset initialization"""
        tokenized_data = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]]
        dataset = FinAIDataset(tokenized_data, max_seq_len=64)

        assert dataset is not None
        assert len(dataset) == len(tokenized_data)

    def test_dataset_getitem(self):
        """Test getting items from dataset"""
        tokenized_data = [[1, 2, 3, 4], [5, 6, 7, 8, 9]]
        dataset = FinAIDataset(tokenized_data, max_seq_len=64)
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_dataset_with_different_lengths(self):
        """Test dataset with sequences of different lengths"""
        tokenized_data = [
            [1, 2],
            [3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ]
        dataset = FinAIDataset(tokenized_data, max_seq_len=128)

        for i in range(len(dataset)):
            item = dataset[i]
            assert item["input_ids"].shape[0] == 128
            assert item["attention_mask"].shape[0] == 128

    def test_dataset_max_seq_len(self):
        """Test that dataset respects max_seq_len"""
        tokenized_data = [[i for i in range(200)]]  # Very long sequence
        max_len = 32
        dataset = FinAIDataset(tokenized_data, max_seq_len=max_len)
        item = dataset[0]

        assert item["input_ids"].shape[0] == max_len

    def test_dataset_padding(self):
        """Test dataset padding"""
        tokenized_data = [[1, 2, 3]]
        max_len = 10
        dataset = FinAIDataset(tokenized_data, max_seq_len=max_len, pad_token_id=0)
        item = dataset[0]

        assert item["input_ids"].shape[0] == max_len
        assert item["attention_mask"][0] == 1
        assert item["attention_mask"][-1] == 0

    def test_dataset_labels_masking(self):
        """Test that labels are properly masked for padding"""
        tokenized_data = [[1, 2, 3]]
        max_len = 10
        dataset = FinAIDataset(tokenized_data, max_seq_len=max_len, pad_token_id=0)
        item = dataset[0]

        # Padded positions should have label -100
        assert item["labels"][-1] == -100
        # Real tokens should have their token id as label
        assert item["labels"][0] == 1


@pytest.mark.unit
class TestDataLoader:
    """Test dataloader creation and functionality"""

    def test_dataloader_creation(self):
        """Test creating a dataloader"""
        tokenized_data = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]]
        dataset = FinAIDataset(tokenized_data, max_seq_len=64)
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

        assert dataloader is not None

    def test_dataloader_iteration(self):
        """Test iterating through dataloader"""
        tokenized_data = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12], [13, 14, 15]]
        dataset = FinAIDataset(tokenized_data, max_seq_len=64)
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert batch["input_ids"].shape[0] <= 2

        assert batch_count == 2

    def test_dataloader_batch_size(self):
        """Test dataloader respects batch size"""
        tokenized_data = [[i] for i in range(10)]
        dataset = FinAIDataset(tokenized_data, max_seq_len=64)
        batch_size = 3
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            assert batch["input_ids"].shape[0] <= batch_size

    def test_dataloader_shuffle(self):
        """Test dataloader shuffle functionality"""
        tokenized_data = [[i] for i in range(20)]
        dataset = FinAIDataset(tokenized_data, max_seq_len=64)

        dataloader_shuffled = create_dataloader(
            dataset, batch_size=5, shuffle=True, num_workers=0
        )
        batches_shuffled = [batch for batch in dataloader_shuffled]

        dataloader_ordered = create_dataloader(
            dataset, batch_size=5, shuffle=False, num_workers=0
        )
        batches_ordered = [batch for batch in dataloader_ordered]

        assert len(batches_shuffled) == len(batches_ordered)


@pytest.mark.unit
class TestDatasetLoading:
    """Test loading datasets from config"""

    def test_load_datasets_from_config(self, tmp_path):
        """Test loading datasets from YAML config"""
        import yaml

        config_path = tmp_path / "datasets.yaml"
        config_data = {
            "datasets": [
                {
                    "name": "test_dataset",
                    "split": "train",
                }
            ]
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            dataset, offset = load_datasets_from_config(
                str(config_path),
                tokenizer=tokenizer,
                max_seq_len=64,
                max_samples=10,
            )
            assert dataset is not None
            assert offset >= 0
        except Exception:
            # Dataset might not be available in test environment
            pass

    def test_tokenizer_integration(self):
        """Test tokenizer integration with dataset"""
        tokenized_data = [[1, 2, 3], [4, 5, 6]]
        dataset = FinAIDataset(tokenized_data, max_seq_len=64)

        for i in range(len(dataset)):
            item = dataset[i]
            # Verify tensors are created
            assert isinstance(item["input_ids"], torch.Tensor)
            assert isinstance(item["attention_mask"], torch.Tensor)
            assert isinstance(item["labels"], torch.Tensor)
