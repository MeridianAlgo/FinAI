"""FinAI configuration"""

from transformers import PretrainedConfig


class FinAIConfig(PretrainedConfig):
    model_type = "finai"

    def __init__(
        self,
        vocab_size=51200,  # Finance-enhanced (gpt2 + ~1k extra)
        hidden_size=1280,
        num_hidden_layers=20,
        num_attention_heads=10, # For GQA (6 query heads, 2 KV heads) - mapping needed
        num_key_value_heads=2,
        intermediate_size=2560,
        hidden_act="swiglu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        # FinAI-Core v2.2 Specifics
        mamba_ratio=0.6,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        ssm_skip_rate=0.45,
        # MoE
        use_moe=True,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=2560,
        # MLA
        mla_latent_rank=64,
        # MTP
        num_mtp_heads=3,
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
        self.ssm_skip_rate = ssm_skip_rate

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

    @classmethod
    def from_yaml(cls, yaml_path):
        import yaml
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        model_config = config.get("model", {})
        return cls(**model_config)
