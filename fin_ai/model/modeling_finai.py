"""
Fin.AI Transformer V3 - Optimized for Performance and Integration
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .configuration_finai import FinAIConfig
except ImportError:
    from configuration_finai import FinAIConfig


class FinAIRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class FinAIRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # cos/sin come from `FinAIRotaryEmbedding` with shape [seq_len, head_dim].
    # Index by `position_ids` ([batch_size, seq_len]) to get [batch_size, seq_len, head_dim],
    # then add a singleton head dimension for broadcasting with q/k: [batch_size, n_heads, seq_len, head_dim].
    cos = cos[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FinAIAttention(nn.Module):
    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.max_seq_len = config.max_seq_len
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.embed_dim, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.embed_dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.embed_dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.embed_dim, bias=False
        )

        self.rotary_emb = FinAIRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_seq_len,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if position_ids is None:
            past_len = 0
            if past_key_value is not None and past_key_value[0] is not None:
                past_len = past_key_value[0].shape[-2]
            position_ids = torch.arange(
                past_len,
                past_len + q_len,
                dtype=torch.long,
                device=hidden_states.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        # Don't reshape attention_mask here - we'll handle it after we know kv_seq_len

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and past_key_value[0] is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # Handle attention mask shape for SDPA vs manual attention
        if attention_mask is not None:
            # First, ensure mask covers kv_seq_len (including past)
            if attention_mask.dim() == 2:
                # [batch, seq] - expand to cover past positions if needed
                original_mask_len = attention_mask.shape[-1]
                if kv_seq_len > original_mask_len:
                    # Pad with True/1.0 for past positions (they're already computed)
                    if attention_mask.dtype == torch.bool:
                        padding = torch.ones(
                            (attention_mask.shape[0], kv_seq_len - original_mask_len),
                            dtype=torch.bool,
                            device=attention_mask.device,
                        )
                        attention_mask = torch.cat([padding, attention_mask], dim=-1)
                    else:
                        padding = torch.zeros(
                            (attention_mask.shape[0], kv_seq_len - original_mask_len),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        attention_mask = torch.cat([padding, attention_mask], dim=-1)
                elif kv_seq_len < original_mask_len:
                    # Truncate mask if needed (shouldn't happen normally)
                    attention_mask = attention_mask[:, -kv_seq_len:]

            # Reshape mask based on attention type
            if self.config.use_flash_attention and not output_attentions:
                # For SDPA: mask must be [batch, q_len, kv_len]
                # Normalize to 2D first if needed
                while attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze(1)
                # Now should be [batch, kv_len] or [batch, q_len_in, kv_len]
                if attention_mask.dim() == 2:
                    # [batch, kv_len] -> [batch, q_len, kv_len]
                    attention_mask = attention_mask.unsqueeze(1).expand(bsz, q_len, -1)
                elif attention_mask.dim() == 3:
                    # [batch, q_len_in, kv_len] - take first q_len rows
                    attention_mask = attention_mask[:, :q_len, :kv_seq_len]
                    # If q_len_in < q_len, repeat last row
                    if attention_mask.shape[1] < q_len:
                        last_row = attention_mask[:, -1:, :]
                        padding = last_row.repeat(1, q_len - attention_mask.shape[1], 1)
                        attention_mask = torch.cat([attention_mask, padding], dim=1)
                # Final check: ensure exact shape
                if attention_mask.shape != (bsz, q_len, kv_seq_len):
                    attention_mask = attention_mask[:, :q_len, :kv_seq_len]
                    if attention_mask.shape[1] < q_len:
                        attention_mask = attention_mask.repeat(
                            1,
                            (q_len + attention_mask.shape[1] - 1)
                            // attention_mask.shape[1],
                            1,
                        )[:, :q_len, :]
                    if attention_mask.shape[2] < kv_seq_len:
                        attention_mask = attention_mask.repeat(
                            1,
                            1,
                            (kv_seq_len + attention_mask.shape[2] - 1)
                            // attention_mask.shape[2],
                        )[:, :, :kv_seq_len]
                # Convert to bool for SDPA
                if attention_mask.dtype != torch.bool:
                    attention_mask = attention_mask.to(torch.bool)
            else:
                # For manual attention: [batch, 1, q_len, kv_len] or [batch, num_heads, q_len, kv_len]
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, None, None, :].expand(
                        bsz, 1, q_len, kv_seq_len
                    )
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask[:, None, :q_len, :kv_seq_len]
                elif attention_mask.dim() == 4:
                    attention_mask = attention_mask[:, :, :q_len, :kv_seq_len]
                # Convert to additive mask if boolean
                if attention_mask.dtype == torch.bool:
                    # Boolean mask: pad with True (allow attention to past)
                    padding = torch.ones(
                        attention_mask.shape[:-1] + (kv_seq_len - original_mask_len,),
                        dtype=torch.bool,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([padding, attention_mask], dim=-1)
                else:
                    # Additive mask: pad with 0.0 (no penalty for past positions)
                    padding = torch.zeros(
                        attention_mask.shape[:-1] + (kv_seq_len - original_mask_len,),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([padding, attention_mask], dim=-1)

        rotary_seq_len = kv_seq_len
        if position_ids is not None:
            # Generation can surface edge cases where the cache length and the provided
            # `position_ids` get temporarily out of sync (e.g. empty caches or alternate
            # cache containers). Make RoPE robust by ensuring the cache covers the
            # maximum position id we will index.
            rotary_seq_len = max(rotary_seq_len, int(position_ids.max()) + 1)

        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None and past_key_value[0] is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat K/V heads for GQA to match query heads
        # Shapes: query [batch, n_heads, q_len, head_dim], key/value [batch, n_kv_heads, kv_len, head_dim]
        if key_states.size(1) != query_states.size(1):
            repeat_factor = query_states.size(1) // key_states.size(1)
            if repeat_factor > 1:
                # Use repeat_interleave to match heads (proper GQA expansion)
                key_states = torch.repeat_interleave(key_states, repeat_factor, dim=1)
            else:
                # This shouldn't happen, but handle it gracefully
                raise ValueError(
                    f"Key heads ({key_states.size(1)}) must be <= query heads ({query_states.size(1)})"
                )

        if value_states.size(1) != query_states.size(1):
            repeat_factor = query_states.size(1) // value_states.size(1)
            if repeat_factor > 1:
                # Use repeat_interleave to match heads (proper GQA expansion)
                value_states = torch.repeat_interleave(
                    value_states, repeat_factor, dim=1
                )
            else:
                raise ValueError(
                    f"Value heads ({value_states.size(1)}) must be <= query heads ({query_states.size(1)})"
                )

        # Ensure all have the same number of heads
        assert (
            query_states.size(1) == key_states.size(1) == value_states.size(1)
        ), f"Head mismatch: q={query_states.size(1)}, k={key_states.size(1)}, v={value_states.size(1)}"

        # Decide whether to use causal masking
        # In generation (q_len=1), is_causal should be False because we only have one query
        # that should attend to everything in the past KV cache.
        # In training/initial pass (q_len > 1), we need causal masking.
        is_causal_processing = self.is_causal and q_len > 1 and past_key_value is None

        # Prepare attention mask
        # If attention_mask is provided (from dataset), it is likely [batch, seq_len] with 1=keep, 0=mask
        # We need to process it to be suitable for attention

        # Flash Attention / SDPA
        if self.config.use_flash_attention and not output_attentions:
            sdp_mask = None
            use_causal = False

            if attention_mask is not None:
                # Prepare padding mask for SDPA
                # Try to create proper mask shape [batch, q_len, kv_len]
                try:
                    if attention_mask.dim() == 2:
                        # [batch, kv_len] -> [batch, q_len, kv_len]
                        sdp_mask = attention_mask.unsqueeze(1).expand(
                            bsz, q_len, kv_seq_len
                        )
                    elif attention_mask.dim() == 3:
                        # [batch, q_len, kv_len] or [batch, n_heads, kv_len] ??
                        # Assuming [batch, q_len_in, kv_len]
                        if (
                            attention_mask.shape[1] == q_len
                            and attention_mask.shape[2] == kv_seq_len
                        ):
                            sdp_mask = attention_mask
                        else:
                            # Reshape to match
                            # (Logic omitted for brevity, fallback to simplest case usually works)
                            sdp_mask = attention_mask[:, :q_len, :kv_seq_len]

                    # If we need causal masking AND have a padding mask, we must Combine them
                    if is_causal_processing:
                        # Create causal mask
                        causal_mask = torch.triu(
                            torch.ones(
                                (q_len, kv_seq_len),
                                dtype=torch.bool,
                                device=attention_mask.device,
                            ),
                            diagonal=1,
                        )
                        # Combine: Mask if (PaddingMask == 0) OR (CausalMask == 1)
                        # If sdp_mask is 1 for keep, 0 for discard
                        # We want sdp_mask to be True for "keep", False for "discard"?
                        # PyTorch SDPA attn_mask:
                        # - binary mask: True ok, False mask?? OR
                        # - additive mask: 0 ok, -inf mask
                        # Docs say: "For a boolean mask, a True indicates that the element should take part in attention."

                        # So: sdp_mask (padding) should be 1/True for keep.
                        # Causal mask: we want to KEEP lower triangle.
                        # So CausalMask (LowerTri) should be True.

                        lower_tri = torch.tril(
                            torch.ones(
                                (q_len, kv_seq_len),
                                dtype=torch.bool,
                                device=attention_mask.device,
                            )
                        )

                        # Convert padding mask to bool if not already
                        if sdp_mask is not None:
                            mask_bool = sdp_mask > 0.5
                            # Final mask = Padding(True) AND Causal(True)
                            sdp_mask = mask_bool & lower_tri
                        else:
                            sdp_mask = lower_tri

                    # Convert to float for additive mask if needed, but SDPA supports bool
                    # However, if sdp_mask is not None, use_causal MUST be False for SDPA in explicit mode
                    pass

                except Exception:
                    sdp_mask = None

            # If no custom mask constructed, use built-in causal capability
            if sdp_mask is None and is_causal_processing:
                use_causal = True

            sdpa_attn_mask = None
            if sdp_mask is not None:
                # Ensure 4D for SDPA if needed [batch, heads, q, k] or [batch, 1, q, k]
                if sdp_mask.dim() == 3:
                    sdpa_attn_mask = sdp_mask.unsqueeze(1)
                else:
                    sdpa_attn_mask = sdp_mask

                # Convert to float mask if it's not bool?
                # PyTorch SDPA handles bool mask (True = keep).
                if sdpa_attn_mask.dtype != torch.bool:
                    sdpa_attn_mask = sdpa_attn_mask > 0.5

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=sdpa_attn_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=use_causal,
            )
            attn_weights = None
        else:
            # Fallback / Manual Attention
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            # Apply Causal Mask
            if is_causal_processing:
                causal_mask = torch.triu(
                    torch.ones(
                        q_len, kv_seq_len, dtype=torch.bool, device=hidden_states.device
                    ),
                    diagonal=1,
                )
                attn_weights.masked_fill_(
                    causal_mask, torch.finfo(attn_weights.dtype).min
                )

            # Apply Padding Mask
            if attention_mask is not None:
                # attention_mask from dataset is [batch, seq_len] with 1=valid, 0=pad
                # We need to mask positions where attention_mask is 0

                # Expand mask to [batch, 1, 1, seq_len] for broadcasting
                if attention_mask.dim() == 2:
                    expanded_mask = attention_mask[:, None, None, :].expand(
                        bsz, 1, q_len, kv_seq_len
                    )
                elif attention_mask.dim() == 3:
                    expanded_mask = attention_mask[:, None, :q_len, :kv_seq_len]
                else:  # 4D
                    expanded_mask = attention_mask

                # Create additive mask: 0.0 for valid, -inf for pad
                # (1.0 - expanded_mask) * min_float

                # Ensure it matches kv_seq_len (past cache handling)
                if expanded_mask.shape[-1] != kv_seq_len:
                    # This usually happens if we have past_key_value
                    # The mask usually covers the full context if generated properly
                    # But for now let's assume specific handling if shapes mismatch
                    pass

                if expanded_mask.dtype == torch.bool:
                    expanded_mask = expanded_mask.to(attn_weights.dtype)

                inverted_mask = 1.0 - expanded_mask
                attn_weights = attn_weights.masked_fill(
                    inverted_mask.bool(), torch.finfo(attn_weights.dtype).min
                )

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            attn_weights = nn.functional.dropout(
                attn_weights, p=self.config.attention_dropout, training=self.training
            )

            if value_states.dim() == 3:
                value_states = value_states.unsqueeze(1)

            if value_states.size(1) != attn_weights.size(1):
                if value_states.size(1) == 1:
                    value_states = value_states.expand(-1, attn_weights.size(1), -1, -1)
                else:
                    repeat_factor = attn_weights.size(1) // value_states.size(1)
                    value_states = torch.repeat_interleave(
                        value_states, repeat_factor, dim=1
                    )
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.config.embed_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value


class FinAIMLP(nn.Module):
    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.embed_dim, config.ff_dim, bias=False)
        self.w2 = nn.Linear(config.ff_dim, config.embed_dim, bias=False)
        self.w3 = nn.Linear(config.embed_dim, config.ff_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FinAIBlock(nn.Module):
    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.attention = FinAIAttention(config)
        self.feed_forward = FinAIMLP(config)
        self.attention_norm = FinAIRMSNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.ffn_norm = FinAIRMSNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)

        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        # `FinAIAttention` returns: (attn_output, attn_weights_or_None, past_key_value_or_None)
        # Make the block output match HF conventions:
        # - if output_attentions=False, return (hidden_states, past_key_value) when use_cache else (hidden_states,)
        # - if output_attentions=True,  return (hidden_states, attn_weights, past_key_value) when use_cache else (hidden_states, attn_weights)
        if output_attentions:
            attn_weights = outputs[0]
            past = outputs[1]
            outputs = (
                (hidden_states, attn_weights, past)
                if use_cache
                else (hidden_states, attn_weights)
            )
        else:
            past = outputs[1]
            outputs = (hidden_states, past) if use_cache else (hidden_states,)

        return outputs


class FinAIPreTrainedModel(PreTrainedModel):
    config_class = FinAIConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FinAIBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def post_init(self):
        # Let PreTrainedModel handle weight init / tie weights first.
        super().post_init()
        # Transformers' GenerationMixin expects a non-None `generation_config`.
        # (CI was failing with: AttributeError: 'NoneType' object has no attribute '_from_model_config')
        if getattr(self, "generation_config", None) is None:
            self.generation_config = GenerationConfig.from_model_config(self.config)


class FinAIModel(FinAIPreTrainedModel, GenerationMixin):
    def __init__(self, config: FinAIConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList(
            [FinAIBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = FinAIRMSNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FinAIModel):
            module.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training and not use_cache:

                def custom_forward(*inputs):
                    return layer(
                        inputs[0],
                        attention_mask=inputs[1],
                        position_ids=inputs[2],
                        past_key_value=None,
                        output_attentions=output_attentions,
                        use_cache=False,
                    )

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class FinAIForCausalLM(FinAIPreTrainedModel, GenerationMixin):
    def __init__(self, config: FinAIConfig):
        super().__init__(config)
        self.model = FinAIModel(config)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Important: Don't manually tie weights here, let post_init handle it
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, recompute_mapping=True, missing_keys=None):
        """Tie the weights between the input embeddings and the output embeddings."""
        if self.config.tie_word_embeddings:
            # Use parent class method to tie weights
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                output_embeddings.weight = input_embeddings.weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """Prepare inputs for generation"""
        has_past = past_key_values is not None and len(past_key_values) > 0

        # Only use the last token if we actually have a cache to append to.
        # Some generation flows may pass an empty tuple for `past_key_values`.
        if has_past:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if has_past:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search"""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
