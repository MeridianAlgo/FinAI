"""Replay Buffer for Continual Learning"""

import random
import json
from collections import deque

class ReplayBuffer:
    def __init__(self, max_tokens=20_000_000):
        self.max_tokens = max_tokens
        self.buffer = deque()
        self.current_tokens = 0

    def add(self, tokens):
        """Add tokens (list of ids) to buffer"""
        token_len = len(tokens)
        while self.current_tokens + token_len > self.max_tokens and self.buffer:
            removed = self.buffer.popleft()
            self.current_tokens -= len(removed)

        self.buffer.append(tokens)
        self.current_tokens += token_len

    def sample(self, num_samples):
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(num_samples, len(self.buffer)))

    def save(self, path):
        with open(path, "w") as f:
            for item in self.buffer:
                f.write(json.dumps(item) + "\n")

    def load(self, path):
        self.buffer.clear()
        self.current_tokens = 0
        try:
            with open(path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    self.add(item)
        except FileNotFoundError:
            pass
