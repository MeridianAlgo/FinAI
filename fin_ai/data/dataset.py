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
        max_seq_len: int = 512,
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


def get_todays_dataset(datasets: List[Dict]) -> tuple[Dict, int]:
    """Get the dataset based on current hour (cycles through datasets every hour)."""
    if not datasets:
        return None, 0
    
    # Get current hour (0-23)
    current_hour = datetime.now().hour
    
    # Cycle through datasets based on hour
    dataset_idx = current_hour % len(datasets)
    
    selected = datasets[dataset_idx]
    
    print(f"📅 Hour: {current_hour:02d}:00 → Dataset #{dataset_idx + 1}/{len(datasets)}: {selected['name']}")
    
    return selected, dataset_idx


def extract_text_from_item(item: Dict, text_column: str) -> Optional[str]:
    """Extract text from item, handling nested fields like 'answers.text'."""
    if '.' in text_column:
        # Handle nested fields
        parts = text_column.split('.')
        value = item
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, "")
            elif isinstance(value, list) and value:
                # Take first item from list
                value = value[0] if value else ""
                if isinstance(value, dict):
                    value = value.get(part, "")
            else:
                return ""
        
        # Handle list results
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        
        return str(value) if value else ""
    else:
        # Simple field access
        value = item.get(text_column, "")
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        return str(value) if value else ""


def load_datasets_from_config(
    config_path: str,
    tokenizer: Optional[AutoTokenizer] = None,
    max_seq_len: int = 512,
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
    ds_config, dataset_idx = get_todays_dataset(datasets)
    
    if not ds_config:
        raise ValueError("No dataset configured!")
    
    name = ds_config["name"]
    subset = ds_config.get("subset")
    split = ds_config.get("split", "train")
    text_column = ds_config.get("text_column", "text")
    ds_max_samples = ds_config.get("max_samples")
    use_streaming = ds_config.get("streaming", False)
    
    print(f"📚 Loading: {name}")
    if use_streaming:
        print("🌊 Using streaming mode (memory efficient)")
    
    try:
        # Force streaming for large datasets to save disk space
        # Always use streaming in CI environments
        import os
        is_ci = os.environ.get("CI", "false").lower() == "true" or os.environ.get("GITHUB_ACTIONS", "false").lower() == "true"
        force_streaming = is_ci or use_streaming or ds_max_samples is None or ds_max_samples > 10000
        
        # Load dataset with streaming for large datasets
        # Use trust_remote_code=True for datasets that need it
        load_kwargs = {
            "split": split,
            "streaming": force_streaming,
            "trust_remote_code": True  # Required for some datasets
        }
        
        if subset:
            dataset = load_dataset(name, subset, **load_kwargs)
        else:
            dataset = load_dataset(name, **load_kwargs)
        
        texts = []
        sample_count = 0
        max_to_load = min(ds_max_samples or 10000, 10000)  # Cap at 10k samples to save space
        
        # Handle streaming vs regular datasets
        if force_streaming:
            # For streaming, iterate and take only what we need
            print(f"🌊 Streaming mode: loading up to {max_to_load:,} samples")
            for item in dataset:
                if sample_count >= max_to_load:
                    break
                text = extract_text_from_item(item, text_column)
                if text and len(str(text).strip()) > 10:
                    texts.append(str(text))
                    sample_count += 1
        else:
            # For regular datasets, select range first
            if ds_max_samples:
                dataset = dataset.select(range(min(ds_max_samples, len(dataset))))
            
            for item in dataset:
                text = extract_text_from_item(item, text_column)
                if text and len(str(text).strip()) > 10:
                    texts.append(str(text))
        
        print(f"📊 Loaded {len(texts):,} samples")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed to load {name}: {error_msg}")
        
        # If this dataset failed, try to fall back to a safe default
        if "Dataset scripts are no longer supported" in error_msg or "trust_remote_code" in error_msg:
            print(f"⚠️  Dataset {name} uses legacy format or requires trust_remote_code")
            print(f"🔄 Falling back to wikitext-2 as safe alternative...")
            
            # Fall back to wikitext-2 (always works)
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                texts = []
                for item in dataset:
                    text = item.get("text", "")
                    if text and len(str(text).strip()) > 10:
                        texts.append(str(text))
                        if len(texts) >= 10000:  # Limit fallback
                            break
                print(f"✅ Fallback successful: {len(texts):,} samples from wikitext-2")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                raise
        else:
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
    
    # Tokenize each text individually to avoid memory issues
    all_tokens = []
    eos_token_id = tokenizer.eos_token_id
    
    for text in texts[:10000]:  # Increased limit for better training
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_seq_len * 2)
            if len(tokens) > 0:
                all_tokens.extend(tokens)
                if eos_token_id:
                    all_tokens.append(eos_token_id)
        except Exception as e:
            continue
    
    # Create fixed-length chunks with stride for more data
    chunks = []
    stride = max_seq_len // 2  # 50% overlap for more training examples
    for i in range(0, len(all_tokens) - max_seq_len + 1, stride):
        chunk = all_tokens[i:i + max_seq_len]
        if len(chunk) == max_seq_len:  # Only use complete chunks
            chunks.append(chunk)
    
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
