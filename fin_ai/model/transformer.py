"""Simplified Fin.AI transformer model"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import json
import os

from fin_ai.model.config import FinAIConfig


class FinAIModel(nn.Module):
    """Simplified GPT-style transformer for Fin.AI"""
    
    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.config = config
        
        # Use GPT2 architecture from transformers as backbone
        from transformers import GPT2Config, GPT2LMHeadModel
        gpt_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_seq_len,
            n_embd=config.embed_dim,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            n_inner=config.ff_dim,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.attention_dropout,
        )
        self.transformer = GPT2LMHeadModel(gpt_config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            output = self(idx_cond)
            logits = output["logits"][:, -1, :]
            
            if do_sample:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu"):
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)
        config = FinAIConfig(**config_dict)
        model = cls(config)
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        return model
