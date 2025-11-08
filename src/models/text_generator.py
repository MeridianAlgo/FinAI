"""Text generation using language model"""
from typing import List
import numpy as np


class TextGenerator:
    """Generate text using trained language model"""
    
    def __init__(self, model, tokenizer, max_length: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7, top_k: int = 50) -> str:
        """Generate text continuation from prompt"""
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            sequence = [self.tokenizer.word_to_idx[self.tokenizer.PAD_TOKEN]] * (self.max_length - len(tokens)) + tokens
        else:
            sequence = tokens[-self.max_length:]
        
        generated_tokens = tokens.copy()
        
        # Generate tokens one by one
        for _ in range(max_tokens):
            # Predict next token
            next_token = self.model.predict_next(sequence, temperature=temperature, top_k=top_k)
            
            # Stop if END token
            if next_token == self.tokenizer.word_to_idx.get(self.tokenizer.END_TOKEN, -1):
                break
            
            generated_tokens.append(next_token)
            
            # Update sequence
            sequence = sequence[1:] + [next_token]
        
        # Decode to text
        response = self.tokenizer.decode(generated_tokens)
        return response
