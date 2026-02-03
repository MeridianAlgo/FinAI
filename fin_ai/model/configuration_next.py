"""FinAI-Next Configuration"""

from transformers import PretrainedConfig


class FinAINextConfig(PretrainedConfig):
    model_type = "finai_next"

    def __init__(
        self,
        vocab_size=151665,
        hidden_size=1536,   # Elite-Consumer sweet spot
        num_layers=48,      # Balanced depth
        max_position_embeddings=32768,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
        liquid_state_dim=384,
        liquid_memory_size=1024,
        skip_threshold=0.1,
        ternary_bits=1.58,
        use_vision_projector=True,
        use_audio_projector=True,
        dynamic_depth_threshold=0.8,
        num_mtp_heads=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.liquid_state_dim = liquid_state_dim
        self.liquid_memory_size = liquid_memory_size
        self.skip_threshold = skip_threshold
        self.ternary_bits = ternary_bits
        self.use_vision_projector = use_vision_projector
        self.use_audio_projector = use_audio_projector
        self.dynamic_depth_threshold = dynamic_depth_threshold
        self.num_mtp_heads = num_mtp_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
