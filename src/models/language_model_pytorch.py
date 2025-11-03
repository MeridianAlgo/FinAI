"""PyTorch-based decoder-only Transformer (GPT) for local LLM training and generation"""
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_drop = nn.Dropout(dropout)
        # causal mask registered as buffer
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    """Tiny GPT-style decoder-only Transformer for local training"""

    def __init__(self, vocab_size: int, block_size: int = 256, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.1, use_gpu: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'wpe': nn.Embedding(block_size, n_embd),
            'h': nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]),
            'ln_f': nn.LayerNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.device = self._get_device(use_gpu)
        self.to(self.device)
        self.is_trained = False

    def _get_device(self, use_gpu: bool):
        if not use_gpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device()
        except Exception:
            pass
        return torch.device('cpu')

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1,T)
        tok_emb = self.transformer['wte'](idx)                    # (B,T,C)
        pos_emb = self.transformer['wpe'](pos)                    # (1,T,C)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)                                   # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, ix = torch.topk(logits, min(top_k, logits.size(-1)))
                probs = torch.zeros_like(logits).scatter_(1, ix, F.softmax(v, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train()
        return idx

    def train_on_tokens(self, tokens: torch.Tensor, steps: int = 1000, batch_size: int = 64, learning_rate: float = 3e-4):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        n = tokens.numel()
        if n < self.block_size + 1:
            raise ValueError("Not enough tokens for training")
        for step in range(steps):
            ix = torch.randint(0, n - self.block_size - 1, (batch_size,), device=self.device)
            x = torch.stack([tokens[i:i + self.block_size] for i in ix])
            y = torch.stack([tokens[i + 1:i + 1 + self.block_size] for i in ix])
            logits, loss = self(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % max(1, steps // 10) == 0 or step == 0:
                print(f"Step {step+1}/{steps} - loss {loss.item():.4f}")
        self.is_trained = True

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'is_trained': self.is_trained,
        }, path)
        print(f"✓ Model saved to {path}")

    @staticmethod
    def load(path: str, use_gpu: bool = True) -> 'LanguageModel':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        ckpt = torch.load(path, map_location='cpu')
        model = LanguageModel(
            vocab_size=ckpt['vocab_size'],
            block_size=ckpt['block_size'],
            use_gpu=use_gpu,
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.is_trained = ckpt.get('is_trained', True)
        print(f"✓ Model loaded from {path}")
        return model

