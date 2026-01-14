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

        # Set standard Transformers attribute names for compatibility with GenerationMixin
        # These must be actual attributes (not properties) for Transformers' __getattribute__ checks
        self.num_hidden_layers = n_layers
        self.hidden_size = embed_dim
        self.num_attention_heads = n_heads

        super().__init__(**kwargs)

    @property
    def num_parameters(self):
        """Calculate approximate number of parameters based on architecture."""
        # Embedding layer
        embed_params = self.vocab_size * self.embed_dim

        # Output layer (if not tied)
        output_params = (
            0 if self.tie_word_embeddings else self.vocab_size * self.embed_dim
        )

        # Per transformer layer:
        # - Attention: Q, K, V projections + output projection
        # - FFN: 2 linear layers (up and down projection)
        # - Layer norms: 2 per layer
        per_layer_params = (
            # Attention (Q, K, V, O)
            self.embed_dim * self.embed_dim * 4
            +
            # FFN (up + down projection)
            self.embed_dim * self.ff_dim * 2
            +
            # Layer norms (2 per layer, each has 2 * embed_dim for weight and bias)
            self.embed_dim * 4
        )

        transformer_params = per_layer_params * self.n_layers

        # Final layer norm
        final_ln_params = self.embed_dim * 2

        total = embed_params + transformer_params + final_ln_params + output_params
        return int(total)

    @classmethod
    def from_yaml(cls, yaml_path):
        import yaml

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        return cls(**model_config)
