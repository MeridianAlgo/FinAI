"""Dataset loading and preparation"""
from typing import List, Tuple, Iterable, Iterator
import numpy as np
from datasets import load_dataset


class DatasetLoader:
    """Load and prepare training data"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_from_file(self, filepath: str) -> List[str]:
        """Load text data from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        return [text.strip() for text in texts if text.strip()]
    
    def iter_from_file(self, filepath: str):
        """Yield non-empty lines from a local text file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    
    def sample_from_file(self, filepath: str, sample_size: int = 1000) -> List[str]:
        """Return the first N non-empty lines from a local text file."""
        out: List[str] = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(line)
                if len(out) >= sample_size:
                    break
        return out
    
    def load_from_hf(self, dataset_id: str, split: str = "train", config: str = None, text_field: str = None, sample_size: int = None) -> List[str]:
        """Load text data from a Hugging Face dataset.
        If sample_size is provided, only the first N samples will be returned.
        If text_field is None, the first string-typed field found will be used.
        config: Optional config name for datasets with multiple configurations.
        """
        texts: List[str] = []
        # Prefer streaming iteration to avoid loading full dataset in memory
        if config:
            ds_stream = load_dataset(dataset_id, config, split=split, streaming=True)
        else:
            ds_stream = load_dataset(dataset_id, split=split, streaming=True)
        count = 0
        # Auto-detect text field from first example if not provided
        detected_field = text_field
        for ex in ds_stream:
            if detected_field is None:
                # find a string field
                for k, v in ex.items():
                    if isinstance(v, str):
                        detected_field = k
                        break
                if detected_field is None:
                    # no plain string field; try join string fields if any
                    str_fields = [k for k, v in ex.items() if isinstance(v, str)]
                    if str_fields:
                        detected_field = str_fields[0]
            if detected_field is None:
                # fallback: concatenate string-like values
                candidate = " ".join([str(v) for v in ex.values() if isinstance(v, (str, int, float))])
            else:
                candidate = ex.get(detected_field, "")
                if not isinstance(candidate, str):
                    candidate = str(candidate)
            candidate = candidate.strip()
            if candidate:
                texts.append(candidate)
                count += 1
                if sample_size is not None and count >= sample_size:
                    break
        return texts
    
    def iter_from_hf(self, dataset_id: str, split: str = "train", config: str = None, text_field: str = None, sample_size: int = None):
        """Yield texts from a Hugging Face dataset as a generator.
        config: Optional config name for datasets with multiple configurations.
        """
        if config:
            ds_stream = load_dataset(dataset_id, config, split=split, streaming=True)
        else:
            ds_stream = load_dataset(dataset_id, split=split, streaming=True)
        count = 0
        detected_field = text_field
        for ex in ds_stream:
            if detected_field is None:
                for k, v in ex.items():
                    if isinstance(v, str):
                        detected_field = k
                        break
                if detected_field is None:
                    str_fields = [k for k, v in ex.items() if isinstance(v, str)]
                    if str_fields:
                        detected_field = str_fields[0]
            if detected_field is None:
                candidate = " ".join([str(v) for v in ex.values() if isinstance(v, (str, int, float))])
            else:
                candidate = ex.get(detected_field, "")
                if not isinstance(candidate, str):
                    candidate = str(candidate)
            candidate = candidate.strip()
            if candidate:
                yield candidate
                count += 1
                if sample_size is not None and count >= sample_size:
                    break
    
    def prepare_sequences(self, texts: List[str], max_length: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input-output sequences for training"""
        X, y = [], []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            
            # Create sequences of varying lengths
            for i in range(1, len(tokens), max(1, stride)):
                sequence = tokens[:i]
                next_token = tokens[i]
                
                # Pad sequence
                if len(sequence) < max_length:
                    sequence = [self.tokenizer.word_to_idx[self.tokenizer.PAD_TOKEN]] * (max_length - len(sequence)) + sequence
                else:
                    sequence = sequence[-max_length:]
                
                X.append(sequence)
                y.append(next_token)
        
        return np.array(X), np.array(y)

    def prepare_sequences_iter(self, texts: Iterable[str], max_length: int, batch_size: int = 1024, stride: int = 1) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield batches of (X, y) for efficient training without holding all data in memory."""
        X_batch: List[List[int]] = []
        y_batch: List[int] = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            for i in range(1, len(tokens), max(1, stride)):
                sequence = tokens[:i]
                next_token = tokens[i]
                if len(sequence) < max_length:
                    sequence = [self.tokenizer.word_to_idx[self.tokenizer.PAD_TOKEN]] * (max_length - len(sequence)) + sequence
                else:
                    sequence = sequence[-max_length:]
                X_batch.append(sequence)
                y_batch.append(next_token)
                if len(X_batch) >= batch_size:
                    yield np.array(X_batch), np.array(y_batch)
                    X_batch, y_batch = [], []
        if X_batch:
            yield np.array(X_batch), np.array(y_batch)
