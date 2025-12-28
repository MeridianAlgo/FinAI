"""Model configuration for Fin.AI"""

from dataclasses import dataclass
import yaml

SIZE_PRESETS = {
    "tiny": {"n_layers": 6, "n_heads": 6, "embed_dim": 384, "ff_dim": 1536},
    "small": {"n_layers": 8, "n_heads": 8, "embed_dim": 512, "ff_dim": 2048},
    "medium": {"n_layers": 12, "n_heads": 8, "embed_dim": 512, "ff_dim": 2048},
    "large": {"n_layers": 24, "n_heads": 12, "embed_dim": 768, "ff_dim": 3072},
}

@dataclass
class FinAIConfig:
    vocab_size: int = 50257
    n_layers: int = 6
    n_heads: int = 6
    embed_dim: int = 384
    ff_dim: int = 1536
    max_seq_len: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_flash_attention: bool = True
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
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
        }
    
    @property
    def num_parameters(self):
        embed_params = self.vocab_size * self.embed_dim
        pos_params = self.max_seq_len * self.embed_dim if self.pos_encoding == "learned" else 0
        attn_params = 4 * self.embed_dim * self.embed_dim
        ff_params = 2 * self.embed_dim * self.ff_dim
        ln_params = 4 * self.embed_dim
        layer_params = attn_params + ff_params + ln_params
        total = embed_params + pos_params + (self.n_layers * layer_params)
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.embed_dim
        return total
