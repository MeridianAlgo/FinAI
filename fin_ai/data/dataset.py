"""Dataset loading and processing for Fin.AI"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Iterator
import yaml
import logging

logger = logging.getLogger(__name__)


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
        
        # Pad if necessary
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


def load_datasets_from_config(
    config_path: str,
    tokenizer: Optional[AutoTokenizer] = None,
    max_seq_len: int = 1024,
    max_samples: Optional[int] = None,
) -> FinAIDataset:
    """
    Load and process datasets from YAML configuration.
    
    Args:
        config_path: Path to datasets.yaml
        tokenizer: Tokenizer to use (loads from config if not provided)
        max_seq_len: Maximum sequence length
        max_samples: Maximum total samples (for testing)
    
    Returns:
        FinAIDataset ready for training
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    if tokenizer is None:
        tokenizer_config = config.get("tokenizer", {})
        tokenizer_name = tokenizer_config.get("pretrained", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load all datasets
    all_texts = []
    
    for ds_config in config.get("datasets", []):
        name = ds_config["name"]
        subset = ds_config.get("subset")
        split = ds_config.get("split", "train")
        text_column = ds_config.get("text_column", "text")
        ds_max_samples = ds_config.get("max_samples")
        streaming = ds_config.get("streaming", False)
        
        logger.info(f"Loading dataset: {name}")
        
        try:
            if subset:
                dataset = load_dataset(name, subset, split=split, streaming=streaming)
            else:
                dataset = load_dataset(name, split=split, streaming=streaming)
            
            # Handle streaming datasets
            if streaming:
                texts = []
                for i, item in enumerate(dataset):
                    if ds_max_samples and i >= ds_max_samples:
                        break
                    text = item.get(text_column, "")
                    if text and len(text.strip()) > 0:
                        texts.append(text)
            else:
                if ds_max_samples:
                    dataset = dataset.select(range(min(ds_max_samples, len(dataset))))
                texts = [item[text_column] for item in dataset if item.get(text_column)]
            
            all_texts.extend(texts)
            logger.info(f"Loaded {len(texts)} samples from {name}")
            
        except Exception as e:
            logger.warning(f"Failed to load dataset {name}: {e}")
            continue
    
    if not all_texts:
        raise ValueError("No texts loaded from any dataset!")
    
    # Apply global max_samples limit
    if max_samples and len(all_texts) > max_samples:
        all_texts = all_texts[:max_samples]
    
    logger.info(f"Total texts: {len(all_texts)}")
    
    # Tokenize and chunk
    preprocessing = config.get("preprocessing", {})
    concat_texts = preprocessing.get("concat_texts", True)
    min_length = preprocessing.get("min_length", 10)
    
    tokenized_chunks = tokenize_and_chunk(
        texts=all_texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        concat_texts=concat_texts,
        min_length=min_length,
    )
    
    logger.info(f"Created {len(tokenized_chunks)} training sequences")
    
    return FinAIDataset(tokenized_chunks, max_seq_len)


def tokenize_and_chunk(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_seq_len: int = 1024,
    concat_texts: bool = True,
    min_length: int = 10,
) -> List[List[int]]:
    """
    Tokenize texts and create fixed-length chunks.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        max_seq_len: Maximum sequence length
        concat_texts: Whether to concatenate texts before chunking
        min_length: Minimum text length to include
    
    Returns:
        List of token ID lists
    """
    # Filter short texts
    texts = [t for t in texts if len(t.strip()) >= min_length]
    
    if concat_texts:
        # Concatenate all texts with EOS token
        eos_token = tokenizer.eos_token or ""
        full_text = eos_token.join(texts)
        
        # Tokenize in batches to avoid memory issues
        all_tokens = []
        batch_size = 10000
        
        for i in range(0, len(full_text), batch_size * 100):
            chunk = full_text[i:i + batch_size * 100]
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        # Create fixed-length chunks
        chunks = []
        for i in range(0, len(all_tokens) - max_seq_len + 1, max_seq_len):
            chunks.append(all_tokens[i:i + max_seq_len])
        
        # Add remaining tokens if substantial
        if len(all_tokens) % max_seq_len > max_seq_len // 2:
            chunks.append(all_tokens[-(len(all_tokens) % max_seq_len):])
        
        return chunks
    else:
        # Tokenize each text separately
        chunks = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            # Split long texts into chunks
            for i in range(0, len(tokens), max_seq_len):
                chunk = tokens[i:i + max_seq_len]
                if len(chunk) > max_seq_len // 4:  # Keep if at least 25% of max length
                    chunks.append(chunk)
        
        return chunks


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
