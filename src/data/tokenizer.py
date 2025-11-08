"""Tokenizer for text processing"""
import pickle
from typing import List


class Tokenizer:
    """Simple byte-level tokenizer"""

    def __init__(self, vocab_size: int = 259):
        self.vocab_size = 259  # 256 bytes + 3 specials
        self.PAD_TOKEN = "<PAD>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.byte_offset = 3

    def fit(self, texts: List[str]):
        return

    def encode(self, text: str) -> List[int]:
        b = text.encode("utf-8", errors="ignore")
        return [self.bos_id] + [self.byte_offset + byte for byte in b] + [self.eos_id]

    def decode(self, indices: List[int]) -> str:
        bytes_list = []
        for idx in indices:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            byte = idx - self.byte_offset
            if 0 <= byte <= 255:
                bytes_list.append(byte)
        return bytes(bytes_list).decode("utf-8", errors="ignore")

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'Tokenizer':
        with open(path, 'rb') as f:
            return pickle.load(f)
