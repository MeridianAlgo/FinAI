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
        max_seq_len=2048,
        dropout=0.1,
        activation="swiglu",
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        use_flash_attention=True,
        attention_dropout=0.1,
        pos_encoding="rotary",
        tie_word_embeddings=True,
        rope_theta=10000.0,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
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
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        super().__init__(**kwargs)

    @classmethod
    def from_yaml(cls, yaml_path):
        import yaml

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        return cls(**model_config)
