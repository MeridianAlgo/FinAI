
"""FinAI configuration"""

from transformers import PretrainedConfig

class FinAIConfig(PretrainedConfig):
    model_type = "finai"

    def __init__(
        self,
        vocab_size=50257,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        embed_dim=512,
        ff_dim=1792,
        max_seq_len=1024,
        dropout=0.1,
        activation="swiglu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        use_flash_attention=True,
        attention_dropout=0.1,
        pos_encoding="rotary",
        tie_word_embeddings=True,
        rope_theta=10000.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_flash_attention = use_flash_attention
        self.attention_dropout = attention_dropout
        self.pos_encoding = pos_encoding
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        
        super().__init__(**kwargs)
