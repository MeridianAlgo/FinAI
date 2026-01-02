"""Model configuration for Fin.AI"""

from dataclasses import dataclass
import yaml

SIZE_PRESETS = {
    "tiny": {"n_layers": 6, "n_heads": 4, "n_kv_heads": 2, "embed_dim": 256, "ff_dim": 896},
    "small": {"n_layers": 8, "n_heads": 8, "n_kv_heads": 4, "embed_dim": 512, "ff_dim": 1792},
    "medium": {"n_layers": 12, "n_heads": 12, "n_kv_heads": 4, "embed_dim": 768, "ff_dim": 2688},
    "large": {"n_layers": 24, "n_heads": 16, "n_kv_heads": 8, "embed_dim": 1024, "ff_dim": 3584},
}

@dataclass
class FinAIConfig:
    vocab_size: int = 50257
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4  # For Grouped Query Attention
    embed_dim: int = 512
    ff_dim: int = 1792  # ~3.5x embed_dim for SwiGLU
    max_seq_len: int = 1024
    dropout: float = 0.1
    activation: str = "swiglu"
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_flash_attention: bool = False  # Custom attention is already optimized
    attention_dropout: float = 0.1
    pos_encoding: str = "rotary"
    tie_word_embeddings: bool = True
    
    @classmethod
    def from_preset(cls, preset: str, **kwargs):
        if preset not in SIZE_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        config_dict = SIZE_PRESETS[preset].copy()
        config_dict.update(kwargs)
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        model_config = config.get("model", {})
        if "size_preset" in model_config:
            preset = model_config.pop("size_preset")
            return cls.from_preset(preset, **model_config)
        return cls(**model_config)
    
    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "activation": self.activation,
            "attention_dropout": self.attention_dropout,
            "pos_encoding": self.pos_encoding,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
    
    @property
    def num_parameters(self):
        """Estimate parameter count for the v2 architecture"""
        # Token embeddings
        embed_params = self.vocab_size * self.embed_dim
        
        # Per-layer parameters
        # GQA: Q uses all heads, K/V use n_kv_heads
        head_dim = self.embed_dim // self.n_heads
        q_params = self.embed_dim * self.embed_dim
        kv_params = 2 * self.embed_dim * (self.n_kv_heads * head_dim)
        o_params = self.embed_dim * self.embed_dim
        attn_params = q_params + kv_params + o_params
        
        # SwiGLU: 3 projections (w1, w2, w3)
        ff_params = 3 * self.embed_dim * self.ff_dim
        
        # RMSNorm: just scale parameter
        norm_params = 2 * self.embed_dim  # 2 norms per layer
        
        layer_params = attn_params + ff_params + norm_params
        total = embed_params + (self.n_layers * layer_params) + self.embed_dim  # +final norm
        
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.embed_dim
        
        return total
