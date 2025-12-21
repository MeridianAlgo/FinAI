"""Dataset loading and processing for Fin.AI"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
from datetime import datetime
import yaml
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class FinAIDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(
        self,
        tokenized_data: List[List[int]],
        max_seq_len: int = 1024,
    ):
        self.data = tokenized_data
        self.max_seq_len = max_seq_len
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx][:self.max_seq_len]
        
        if len(tokens) < self.max_seq_len:
            padding = [0] * (self.max_seq_len - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * len(padding)
            tokens = tokens + padding
        else:
            attention_mask = [1] * self.max_seq_len
        
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
        }


def get_todays_dataset(datasets: List[Dict]) -> Dict:
    """Get the dataset for today based on day of week (0=Sunday, 6=Saturday)."""
    today = datetime.now().weekday()  # 0=Monday in Python
    # Convert to our format: 0=Sunday, 1=Monday, etc.
    day_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
    our_day = day_map[today]
    
    for ds in datasets:
        if ds.get("day") == our_day:
            return ds
    
    # Fallback to first dataset
    return datasets[0] if datasets else None


def load_datasets_from_config(
    config_path: str,
    tokenizer: Optional[AutoTokenizer] = None,
    max_seq_len: int = 1024,
    max_samples: Optional[int] = None,
) -> FinAIDataset:
    """Load today's dataset from YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if tokenizer is None:
        tokenizer_config = config.get("tokenizer", {})
        tokenizer_name = tokenizer_config.get("pretrained", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, verbose=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Get today's dataset
    datasets = config.get("datasets", [])
    ds_config = get_todays_dataset(datasets)
    
    if not ds_config:
        raise ValueError("No dataset configured for today!")
    
    name = ds_config["name"]
    subset = ds_config.get("subset")
    split = ds_config.get("split", "train")
    text_column = ds_config.get("text_column", "text")
    ds_max_samples = ds_config.get("max_samples")
    
    print(f"📅 Today's dataset: {name}")
    
    try:
        if subset:
            dataset = load_dataset(name, subset, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(name, split=split, trust_remote_code=True)
        
        if ds_max_samples:
            dataset = dataset.select(range(min(ds_max_samples, len(dataset))))
        
        texts = []
        for item in dataset:
            text = item.get(text_column, "")
            if text and len(str(text).strip()) > 10:
                texts.append(str(text))
        
        print(f"📊 Loaded {len(texts):,} samples")
        
    except Exception as e:
        print(f"⚠️ Failed to load {name}: {e}")
        raise
    
    if not texts:
        raise ValueError("No texts loaded!")
    
    if max_samples and len(texts) > max_samples:
        texts = texts[:max_samples]
    
    # Tokenize
    preprocessing = config.get("preprocessing", {})
    min_length = preprocessing.get("min_length", 10)
    
    tokenized_chunks = tokenize_and_chunk(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        min_length=min_length,
    )
    
    print(f"🔢 Created {len(tokenized_chunks):,} training sequences")
    
    return FinAIDataset(tokenized_chunks, max_seq_len)


def tokenize_and_chunk(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_seq_len: int = 512,
    min_length: int = 10,
) -> List[List[int]]:
    """Tokenize texts and create fixed-length chunks."""
    texts = [t for t in texts if len(t.strip()) >= min_length]
    
    # Concatenate all texts
    eos_token = tokenizer.eos_token or ""
    full_text = eos_token.join(texts[:10000])  # Limit to avoid memory issues
    
    # Tokenize
    all_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    
    # Create chunks
    chunks = []
    for i in range(0, len(all_tokens) - max_seq_len + 1, max_seq_len):
        chunks.append(all_tokens[i:i + max_seq_len])
    
    return chunks


def create_dataloader(
    dataset: FinAIDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
