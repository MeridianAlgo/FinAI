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
import json
import os

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


def get_prioritized_datasets(datasets: List[Dict], cycle_config: Optional[Dict] = None) -> List[Dict]:
    """Get list of datasets, starting with today's dataset or next in cycle."""
    if not datasets:
        return []
    
    start_idx = 0
    use_day_rotation = True
    
    # Check if we should use per-run rotation cycle
    if cycle_config and cycle_config.get("rotate_on_run"):
        state_file = cycle_config.get("state_file", "checkpoints/dataset_state.json")
        try:
            current_index = 0
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                    current_index = state.get("last_index", -1) + 1
            
            start_idx = current_index % len(datasets)
            
            # Save new state
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "w") as f:
                json.dump({
                    "last_index": start_idx,
                    "dataset_name": datasets[start_idx]["name"],
                    "updated_at": datetime.now().isoformat()
                }, f, indent=2)
            
            print(f"🔄 Cycle rotation: Selected index {start_idx} ({datasets[start_idx]['name']})")
            use_day_rotation = False
        except Exception as e:
            print(f"⚠️ Failed to manage dataset cycle state: {e}. Falling back to day-based rotation.")
            use_day_rotation = True

    if use_day_rotation:
        today = datetime.now().weekday()  # 0=Monday
        # Convert to our format: 0=Sunday, 1=Monday, etc.
        day_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
        our_day = day_map[today]
        
        # Find today's dataset index
        for i, ds in enumerate(datasets):
            if ds.get("day") == our_day:
                start_idx = i
                break
        print(f"📅 Day-based rotation: Selected {datasets[start_idx]['name']} for day {our_day}")
            
    # Rotate list so the selected one is first
    return datasets[start_idx:] + datasets[:start_idx]


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
    
    # Get datasets list & cycle config
    datasets_config_list = config.get("datasets", [])
    cycle_config = config.get("cycle", {})
    prioritized_list = get_prioritized_datasets(datasets_config_list, cycle_config)
    
    if not prioritized_list:
        raise ValueError("No datasets configured!")
    
    # Try datasets in order
    texts = []
    loaded_name = None
    last_error = None
    
    for ds_config in prioritized_list:
        name = ds_config["name"]
        subset = ds_config.get("subset")
        split = ds_config.get("split", "train")
        text_column = ds_config.get("text_column", "text")
        ds_max_samples = ds_config.get("max_samples")
        
        print(f"🔄 Attempting dataset: {name}")
        
        try:
            if subset:
                dataset = load_dataset(name, subset, split=split, trust_remote_code=True)
            else:
                dataset = load_dataset(name, split=split, trust_remote_code=True)
            
            if ds_max_samples:
                dataset = dataset.select(range(min(ds_max_samples, len(dataset))))
            
            # Extract texts
            current_texts = []
            for item in dataset:
                text = item.get(text_column, "")
                if text and len(str(text).strip()) > 10:
                    current_texts.append(str(text))
            
            if current_texts:
                texts = current_texts
                loaded_name = name
                print(f"✅ Successfully loaded {name} with {len(texts):,} samples")
                break
            else:
                print(f"⚠️ Loaded {name} but found no valid texts, trying next...")
                
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")
            last_error = e
            continue
    
    if not texts:
        raise ValueError(f"All datasets failed! Last error: {last_error}")
    
    print(f"📅 Training on: {loaded_name}")
    
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
