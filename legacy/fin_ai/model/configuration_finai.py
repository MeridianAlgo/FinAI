"""FinAI configuration"""

from transformers import PretrainedConfig


class FinAIConfig(PretrainedConfig):
    model_type = "finai"

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=1536,  # Ultra-Lite target
        hidden_act="swiglu",
        max_position_embeddings=8192,  # Default 8k
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        # FinAI-Core v2.2 Ultra-Lite Specifics
        mamba_ratio=0.5,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        ssm_skip_threshold=0.15,  # Heuristic threshold for skipping
        # MoE
        use_moe=True,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=1536,  # Ultra-Lite target
        # MLA
        mla_latent_rank=64,  # Spec rank 48-64
        # MTP
        num_mtp_heads=2,  # 1 main + 1 additional (total 2 in list usually means 1 main + 1 aux, but implementation uses list len for aux)
        mtp_weight=0.5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        self.mamba_ratio = mamba_ratio
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.ssm_skip_threshold = ssm_skip_threshold

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.mla_latent_rank = mla_latent_rank
        self.num_mtp_heads = num_mtp_heads
        self.mtp_weight = mtp_weight

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
