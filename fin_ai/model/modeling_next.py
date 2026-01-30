"""FinAI-Next Modeling"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from .configuration_next import FinAINextConfig
from .bitnet import BitLinear, BitRMSNorm
from .liquid_blocks import LiquidBlock
from .adaptive_compute import AdaptiveComputeWrapper, MultimodalProjector


class FinAINextPreTrainedModel(PreTrainedModel):
    config_class = FinAINextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LiquidBlock):
            module.checkpointing = value


class FinAINextModel(FinAINextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LiquidBlock(config) for _ in range(config.num_layers)])
        self.adaptive_wrappers = nn.ModuleList([AdaptiveComputeWrapper(config, i) for i in range(config.num_layers)])
        self.norm = BitRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_vision_projector:
            self.vision_projector = MultimodalProjector(config, 768, "vision")
        if config.use_audio_projector:
            self.audio_projector = MultimodalProjector(config, 128, "audio")
        self.post_init()

    def forward(self, input_ids, labels=None, vision_features=None, audio_features=None):
        hidden_states = self.embed_tokens(input_ids)
        if vision_features is not None and hasattr(self, "vision_projector"):
            v_proj = self.vision_projector(vision_features)
            hidden_states = hidden_states + v_proj[:, :hidden_states.size(1), :]
        if audio_features is not None and hasattr(self, "audio_projector"):
            a_proj = self.audio_projector(audio_features)
            hidden_states = hidden_states + a_proj[:, :hidden_states.size(1), :]
        liquid_state = None
        for i, layer in enumerate(self.layers):
            hidden_states, liquid_state = layer(hidden_states, liquid_state)
            hidden_states, should_skip = self.adaptive_wrappers[i](hidden_states)
            if should_skip and i > self.config.num_layers // 2:
                break
        hidden_states = self.norm(hidden_states)
        return hidden_states


class FinAINextForCausalLM(FinAINextPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = FinAINextModel(config)
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, value): self.lm_head = value

    def forward(self, input_ids, labels=None, **kwargs):
        hidden_states = self.model(input_ids, **kwargs)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return type("CausalLMOutput", (), {"loss": loss, "logits": logits})
