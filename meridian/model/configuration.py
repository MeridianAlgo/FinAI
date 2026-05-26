"""Meridian.AI Configuration.

Architecture highlights:
 - Sparse Mixture-of-Experts (SMoE) with top-k routing
 - Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE)
 - SwiGLU feed-forward blocks
 - RMSNorm (faster than LayerNorm)
 - Financial Numeracy Encoding (novel)
 - ~300M total params, ~100M active per token
"""

from transformers import PretrainedConfig


class MeridianSMoEConfig(PretrainedConfig):
    """Configuration for Meridian.AI."""

    model_type = "meridian_smoe"

    def __init__(
        self,
        vocab_size: int = 151_665,
        hidden_size: int = 768,
        intermediate_size: int = 1792,  # SwiGLU: ~2.3x hidden
        num_layers: int = 14,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,  # GQA: 4 KV heads shared across 12 Q heads
        max_position_embeddings: int = 2048,
        rope_theta: float = 500_000.0,
        rms_norm_eps: float = 1e-6,
        # --- Mixture of Experts ---
        num_experts: int = 8,
        num_experts_per_token: int = 2,  # top-k routing
        expert_intermediate_size: int = 896,  # Each expert is smaller
        moe_layer_frequency: int = 2,  # Every 2nd layer is MoE (others are dense)
        router_aux_loss_coef: float = 0.01,  # Load-balancing loss
        # --- Financial Numeracy ---
        use_numeracy_encoding: bool = True,
        numeracy_embed_dim: int = 64,
        # --- Training ---
        gradient_checkpointing: bool = True,
        tie_word_embeddings: bool = True,  # Saves ~116M params by sharing embed & lm_head
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 151_643,
        bos_token_id: int = 151_643,
        eos_token_id: int = 151_645,
        # --- Continual Learning ---
        ewc_lambda: float = 100.0,  # Elastic Weight Consolidation strength
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        # MoE
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_intermediate_size = expert_intermediate_size
        self.moe_layer_frequency = moe_layer_frequency
        self.router_aux_loss_coef = router_aux_loss_coef

        # Numeracy
        self.use_numeracy_encoding = use_numeracy_encoding
        self.numeracy_embed_dim = numeracy_embed_dim

        # Training
        self.gradient_checkpointing = gradient_checkpointing
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # Continual Learning
        self.ewc_lambda = ewc_lambda

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


# Backward-compatibility alias
MeridianConfig = MeridianSMoEConfig
