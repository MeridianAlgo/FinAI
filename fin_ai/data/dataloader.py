"""DataLoader utilities for Fin.AI"""

from torch.utils.data import DataLoader
from fin_ai.data.dataset import FinAIDataset


def create_dataloader(
    dataset: FinAIDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
