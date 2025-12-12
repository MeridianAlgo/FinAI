"""
Highly Efficient Transformer Model for CPU Training
Optimized for GitHub Actions with limited compute
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """Efficient RoPE implementation"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        # dim should be head_dim here
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, n_head, T, head_dim)
        seq_len = x.shape[2]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Return (T, head_dim) tensors
        return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings"""
    # x: (B, n_head, T, head_dim)
    # cos, sin: (T, head_dim)
    seq_len = x.shape[2]
    d = x.shape[-1]
    
    # Ensure cos/sin match head_dim
    cos = cos[:seq_len, :d]
    sin = sin[:seq_len, :d]
    
    # Split into two halves
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos_half = cos[..., :d//2]
    sin_half = sin[..., :d//2]
    
    # Broadcast: (T, d//2) -> (1, 1, T, d//2)
    cos_half = cos_half[None, None, :, :]
    sin_half = sin_half[None, None, :, :]
    
    return torch.cat([
        x1 * cos_half - x2 * sin_half,
        x1 * sin_half + x2 * cos_half
    ], dim=-1)


class EfficientAttention(nn.Module):
    """Memory-efficient multi-head attention"""
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5
        
        # Fused QKV projection
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, block_size)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Apply RoPE
        cos, sin = self.rope(x)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Efficient attention with PyTorch's scaled_dot_product_attention
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Fallback manual attention
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
        
        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """SwiGLU activation - more efficient than standard FFN"""
    def __init__(self, n_embd: int, expansion: float = 2.67):
        super().__init__()
        hidden = int(n_embd * expansion)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """RMSNorm - faster than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    """Efficient transformer block"""
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = EfficientAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class EfficientFinAI(nn.Module):
    """
    Efficient Transformer Language Model
    Optimized for CPU training with:
    - RoPE positional encoding
    - SwiGLU activation
    - RMSNorm (faster than LayerNorm)
    - Efficient attention
    - Weight tying
    """
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 6,
        block_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        
        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])
        
        # Final norm and output
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"EfficientFinAI: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Token embeddings
        x = self.tok_emb(idx)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm and logits
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate text"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_model(vocab_size: int, config: dict = None) -> EfficientFinAI:
    """Create model with default or custom config"""
    default_config = {
        'n_embd': 384,
        'n_head': 6,
        'n_layer': 6,
        'block_size': 512,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return EfficientFinAI(vocab_size, **default_config)
