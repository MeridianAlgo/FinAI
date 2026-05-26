"""Meridian.AI — A 300M Sparse MoE Finance LLM.

Novel architecture combining:
 1. Sparse Mixture-of-Experts (SMoE) with load-balanced top-k routing
 2. Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE)
 3. SwiGLU gated FFN
 4. RMSNorm
 5. Financial Numeracy Encoding — inject number-magnitude awareness
 6. Gradient checkpointing for CPU-friendly memory
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from meridian.model.configuration import MeridianConfig, MeridianSMoEConfig

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard implementation with stability cast
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with extended context support."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 500_000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            # Ensure in float32 for cos/sin calculation
            self._cached_cos = emb.cos().float().unsqueeze(0).unsqueeze(0)  # [1,1,S,D]
            self._cached_sin = emb.sin().float().unsqueeze(0).unsqueeze(0)

        cos = self._cached_cos[:, :, :seq_len, :].to(x.dtype)
        sin = self._cached_sin[:, :, :seq_len, :].to(x.dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Attention: Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------


class MeridianAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) + RoPE.

    GQA uses fewer KV heads than Q heads, saving memory & compute.
    With 16 Q heads and 4 KV heads, each KV head serves 4 Q heads.
    """

    def __init__(self, config: MeridianConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj._is_residual = True

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Handle KV cache
        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(q, kv_seq_len)
        if past_key_value is not None:
            # Match current sequence length
            cos = cos[:, :, -q_len:, :]
            sin = sin[:, :, -q_len:, :]

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Repeat KV heads for GQA compatibility with Q heads
        if self.num_kv_groups > 1:
            k = (
                k.unsqueeze(2)
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(bsz, self.num_heads, -1, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(bsz, self.num_heads, -1, self.head_dim)
            )

        # Stable attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in float32 for stability
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), new_kv


# ---------------------------------------------------------------------------
# Feed-Forward: SwiGLU
# ---------------------------------------------------------------------------


class MeridianSwiGLU(nn.Module):
    """SwiGLU feed-forward network — state-of-the-art gated activation.

    SwiGLU(x) = (xW_gate . SiLU(xW_up)) . W_down
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.down_proj._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Mixture of Experts (MoE) with Load-Balanced Routing
# ---------------------------------------------------------------------------


class ExpertRouter(nn.Module):
    """Top-k expert router with auxiliary load-balancing loss."""

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_token: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (router_weights, expert_indices, aux_loss)."""
        # Add small noise during training to prevent routing collapse (jitter)
        if self.training:
            noise = torch.randn_like(hidden_states) * 0.01
            router_logits = self.gate(hidden_states + noise)
        else:
            router_logits = self.gate(hidden_states)

        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        topk_weights, topk_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)

        # Safe normalization for weights
        weights_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / (weights_sum + 1e-8)

        # Load-balancing auxiliary loss (Switch Transformer style)
        # 1. P(e) = Mean probability of choosing expert e
        # 2. f(e) = Fraction of tokens routed to expert e
        # Loss = num_experts * sum(f(e) * P(e))

        # Calculate f(e): fraction of tokens routed to each expert (vectorized)
        f_e = (
            torch.bincount(topk_indices[:, 0], minlength=self.num_experts).float()
            / hidden_states.shape[0]
        )

        # Calculate P(e): mean routing probability
        P_e = router_probs.mean(dim=0)

        aux_loss = self.num_experts * (f_e * P_e).sum()

        return topk_weights, topk_indices, aux_loss


class MeridianMoELayer(nn.Module):
    """Sparse Mixture-of-Experts feed-forward layer."""

    def __init__(self, config: MeridianConfig):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                MeridianSwiGLU(config.hidden_size, config.expert_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.router = ExpertRouter(
            config.hidden_size, config.num_experts, config.num_experts_per_token
        )
        self.num_experts_per_token = config.num_experts_per_token

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, orig_shape[-1])

        router_weights, expert_indices, aux_loss = self.router(flat_hidden)

        # Sparse computation
        output = torch.zeros_like(flat_hidden)

        # Optimize: group tokens by expert to avoid redundant forward passes
        for expert_idx in range(len(self.experts)):
            # Find tokens where this expert is among the top-k
            mask = expert_indices == expert_idx
            if not mask.any():
                continue

            # Identify which tokens (row indices) and which k-slot (col indices)
            row_indices, k_slots = mask.nonzero(as_tuple=True)

            # Run expert once for all unique tokens that need it
            unique_rows = row_indices.unique()
            expert_input = flat_hidden[unique_rows]
            expert_out = self.experts[expert_idx](expert_input)

            # Map each row_index to its position in unique_rows (vectorized, no Python dict)
            mapped_indices = torch.searchsorted(unique_rows.contiguous(), row_indices.contiguous())

            # Collect and add to final output
            w = router_weights[row_indices, k_slots].unsqueeze(-1)
            output.index_add_(0, row_indices, w * expert_out[mapped_indices])

        return output.view(*orig_shape), aux_loss


# ---------------------------------------------------------------------------
# Financial Numeracy Encoding
# ---------------------------------------------------------------------------


class NumeracyEncoder(nn.Module):
    """Magnitude-aware signals for numeric tokens.

    Injects a lightweight learned signal based on the order-of-magnitude bucket
    for each input token. Tokens that represent numbers get a signal indicating
    their approximate magnitude (units, thousands, millions, etc.); non-numeric
    tokens get bucket 0 (a near-zero signal since the embedding is zero-initialized
    and the weight is small).

    Usage:
        # At model init time, optionally call register_digit_tokens() with a tokenizer
        # to populate proper magnitude buckets. Without this call, all tokens map to
        # bucket 0 and the encoder produces no net signal (correct fall-back).
        encoder.register_digit_tokens(tokenizer)
    """

    def __init__(self, hidden_size: int, numeracy_dim: int = 64, vocab_size: int = 151_665):
        super().__init__()
        self.num_buckets = 32
        self.vocab_size = vocab_size
        # Bucket 0 = "non-numeric / unknown", buckets 1-31 = order-of-magnitude bands
        self.magnitude_embed = nn.Embedding(self.num_buckets, numeracy_dim)
        nn.init.zeros_(self.magnitude_embed.weight)  # Zero-init → no signal until trained
        self.proj = nn.Linear(numeracy_dim, hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)  # Zero-init → no distortion at start

        # Static lookup: token_id → magnitude bucket (populated by register_digit_tokens)
        self.register_buffer(
            "magnitude_buckets", torch.zeros(vocab_size, dtype=torch.long)
        )

    def register_digit_tokens(self, tokenizer) -> int:
        """Populate magnitude_buckets from a tokenizer's vocabulary.

        Iterates over all token strings, tries to parse each as a number, and
        assigns it to a magnitude bucket: bucket = clamp(floor(log10(|val|)) + 8, 0, 31).
        Non-numeric tokens stay at bucket 0.

        Returns the number of numeric tokens found.
        """
        import math

        buckets = torch.zeros(self.vocab_size, dtype=torch.long)
        numeric_count = 0

        vocab = tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            if token_id >= self.vocab_size:
                continue
            # Strip BPE prefix artifacts (▁, Ġ, etc.) and whitespace
            clean = token_str.lstrip("▁Ġ").strip()
            try:
                val = float(clean)
                if val != 0:
                    magnitude = int(math.log10(abs(val)))  # e.g. 1→0, 10→1, 1000→3
                else:
                    magnitude = -8  # zero gets bucket 0
                bucket = max(0, min(self.num_buckets - 1, magnitude + 8))
                buckets[token_id] = bucket
                numeric_count += 1
            except (ValueError, TypeError, OverflowError):
                pass  # Non-numeric token → stays at bucket 0

        self.magnitude_buckets = buckets
        return numeric_count

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Clamp to valid range (guard against out-of-vocab IDs)
        ids = input_ids.clamp(0, self.vocab_size - 1)
        bucket_ids = self.magnitude_buckets[ids]
        numeracy_emb = self.magnitude_embed(bucket_ids)
        numeracy_signal = self.proj(numeracy_emb)
        return hidden_states + 0.05 * numeracy_signal


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class MeridianDecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: MeridianConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = MeridianAttention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Alternate dense/MoE layers
        self.is_moe = layer_idx % config.moe_layer_frequency == 1
        if self.is_moe:
            self.moe = MeridianMoELayer(config)
        else:
            self.ffn = MeridianSwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN (dense or MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.is_moe:
            hidden_states, aux_loss = self.moe(hidden_states)
        else:
            hidden_states = self.ffn(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states, new_kv, aux_loss


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class MeridianModel(nn.Module):
    """Meridian.AI backbone."""

    def __init__(self, config: MeridianConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [MeridianDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.numeracy = None
        if config.use_numeracy_encoding:
            self.numeracy = NumeracyEncoder(
                config.hidden_size, config.numeracy_embed_dim, config.vocab_size
            )

        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, list, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)

        if self.numeracy is not None:
            hidden_states = self.numeracy(hidden_states, input_ids)

        # Causal mask construction
        bsz, seq_len = input_ids.shape
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        if attention_mask is not None:
            # Avoid NaN by multiplying 0.0 with -inf. Use finfo.min instead, or just boolean masking.
            inv_mask = 1.0 - attention_mask.to(hidden_states.dtype)
            mask_value = torch.finfo(hidden_states.dtype).min
            padding_mask = inv_mask * mask_value
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask.unsqueeze(
                1
            ).unsqueeze(2)
        else:
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        all_kvs = []
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                hidden_states, new_kv, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    combined_mask,
                    past_kv,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, new_kv, aux_loss = layer(
                    hidden_states, combined_mask, past_kv, use_cache
                )

            all_kvs.append(new_kv)
            total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_kvs, total_aux_loss


class MeridianSMoEForCausalLM(PreTrainedModel):
    """Meridian.AI for causal language modeling."""

    config_class = MeridianSMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # Default: no class-level tied keys. Set dynamically in __init__ based on config.
    # transformers>=5.0 expects _tied_weights_keys as a dict {tied_key: source_key}
    _tied_weights_keys: dict = {}

    def __init__(self, config: MeridianSMoEConfig):
        super().__init__(config)
        self.config = config
        self.model = MeridianModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Set weight tying keys dynamically — only when tie_word_embeddings=True
        if config.tie_word_embeddings:
            self._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        else:
            self._tied_weights_keys = {}

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Improved weight initialization for stable training.

        Uses torch.nn.init.* functions (not tensor.data.normal_()) so that
        transformers>=5.0's @init.guard_torch_init_functions() decorator can
        correctly mark initialized tensors and skip re-initialization after
        loading from a checkpoint.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # Normal distribution based on Xavier/Kaiming principles
            if getattr(module, "_is_residual", False):
                nn.init.normal_(
                    module.weight, mean=0.0, std=std / math.sqrt(2 * self.config.num_layers)
                )
            else:
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, past_key_values_out, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal training
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Use float32 for cross entropy stability
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size).to(torch.float32),
                shift_labels.view(-1),
                ignore_index=-100,
            ).to(logits.dtype)

            # Add auxiliary MoE loss
            loss = loss + (self.config.router_aux_loss_coef * aux_loss.to(loss.dtype))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values_out if use_cache else None,
        )

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        past_key_values = None
        generated = input_ids

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            outputs = self.forward(curr_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                indices_to_remove = sorted_indices[remove_mask]
                next_logits[:, indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.config.eos_token_id:
                break

        return generated


# Backward-compatibility alias
MeridianForCausalLM = MeridianSMoEForCausalLM
