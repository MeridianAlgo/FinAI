"""
FinAI-Core v2.2 Ultra-Lite - High-Efficiency Financial Model
Hybrid Mamba-2 SSM + Transformer with MoE, MLA, MTP and Delta-RoPE
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_finai import FinAIConfig


class FinAIRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class DeltaRoPE(nn.Module):
    """Gated MLP that learns delta updates to rotary frequencies"""

    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.gate = nn.Linear(config.hidden_size, 1)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, self.dim // 4),
            nn.SiLU(),
            nn.Linear(self.dim // 4, self.dim),
        )

    def forward(self, hidden_states, inv_freq):
        # hidden_states: [bs, seq, dim]
        # inv_freq: [head_dim // 2]
        gate = torch.sigmoid(
            self.gate(hidden_states.mean(dim=1, keepdim=True))
        )  # [bs, 1, 1]
        delta = self.mlp(hidden_states.mean(dim=1, keepdim=True))  # [bs, 1, head_dim]
        # Only affect the frequencies (real/imag parts)
        delta_freq = delta[..., : inv_freq.shape[0]]
        return inv_freq + gate * delta_freq


class FinAIRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len, delta_inv_freq=None):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)

        freqs_base = self.inv_freq
        if delta_inv_freq is not None:
            # delta_inv_freq: [bs, 1, head_dim // 2]
            freqs = torch.einsum("i, bnj -> bnij", t, delta_inv_freq)
        else:
            freqs = torch.outer(t, freqs_base)

        emb = torch.cat((freqs, freqs), dim=-1)
        if emb.dim() == 2:  # [seq, head_dim]
            return emb.cos(), emb.sin()
        else:  # [bs, 1, seq, head_dim]
            return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Adjust shapes for broadcasting
    if cos.dim() == 2:  # [seq, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:  # [bs, 1, seq, dim]
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Mamba2Block(nn.Module):
    """Simplified Mamba-2 SSM block with Sparse Recurrent Skipping"""

    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.d_state = config.mamba_d_state
        self.d_conv = config.mamba_d_conv
        self.expand = config.mamba_expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.config.mamba_d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Sparse skipping heuristic: small head to predict token importance
        self.skip_heuristic = nn.Linear(self.d_model, 1)
        self.skip_threshold = config.ssm_skip_threshold

    def forward(self, x):
        batch, seqlen, dim = x.shape

        # Token-wise skipping heuristic
        importance = torch.sigmoid(self.skip_heuristic(x))  # [bs, seq, 1]
        mask = (importance > self.skip_threshold).float()

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.transpose(1, 2)
        x_inner = self.conv1d(x_inner)[:, :, :seqlen]
        x_inner = x_inner.transpose(1, 2)

        x_inner = F.silu(x_inner)

        x_db = self.x_proj(x_inner)
        dt, B, C = torch.split(x_db, [1, self.d_state // 2, self.d_state // 2], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # Apply skipping: only update where mask is 1
        y = x_inner * torch.tanh(dt) * mask
        return self.out_proj(y * F.silu(z))


class MLAAttention(nn.Module):
    """Multi-head Latent Attention with Delta-RoPE"""

    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.latent_rank = config.mla_latent_rank
        self.head_dim = self.hidden_size // self.num_heads

        self.q_latent_proj = nn.Linear(self.hidden_size, self.latent_rank, bias=False)
        self.q_heads_proj = nn.Linear(
            self.latent_rank, self.num_heads * self.head_dim, bias=False
        )

        self.kv_latent_proj = nn.Linear(self.hidden_size, self.latent_rank, bias=False)
        self.kv_heads_proj = nn.Linear(
            self.latent_rank, self.num_heads * self.head_dim * 2, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = FinAIRotaryEmbedding(self.head_dim)
        self.delta_rope = DeltaRoPE(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, q_len, _ = hidden_states.size()

        q_latent = self.q_latent_proj(hidden_states)
        q = (
            self.q_heads_proj(q_latent)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_latent = self.kv_latent_proj(hidden_states)
        kv = (
            self.kv_heads_proj(kv_latent)
            .view(bsz, q_len, self.num_heads, self.head_dim * 2)
            .transpose(1, 2)
        )
        k, v = kv.chunk(2, dim=-1)

        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)

        # Delta-RoPE frequencies
        delta_inv_freq = self.delta_rope(hidden_states, self.rotary_emb.inv_freq)
        cos, sin = self.rotary_emb(v, seq_len=q_len, delta_inv_freq=delta_inv_freq)

        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)

        return self.o_proj(attn_output)


class DeepSeekMoE(nn.Module):
    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.intermediate_size, self.hidden_size, bias=False),
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x):
        bsz, seq_len, h = x.shape
        x_flat = x.view(-1, h)
        logits = self.gate(x_flat)
        weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                matches = top_indices[mask] == i
                weight = top_weights[mask][matches].unsqueeze(-1)
                out[mask] += expert_out * weight

        return out.view(bsz, seq_len, h)


class FinAIBlock(nn.Module):
    def __init__(self, config: FinAIConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_mamba = (layer_idx / config.num_hidden_layers) < config.mamba_ratio

        self.pre_norm = FinAIRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.is_mamba:
            self.mamba = Mamba2Block(config)
        else:
            self.attn = MLAAttention(config)

        self.post_norm = FinAIRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.use_moe:
            self.moe = DeepSeekMoE(config)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
            )

    def forward(self, x, attention_mask=None, position_ids=None):
        residual = x
        x = self.pre_norm(x)
        if self.is_mamba:
            x = self.mamba(x)
        else:
            x = self.attn(x, attention_mask, position_ids)
        x = residual + x

        residual = x
        x = self.post_norm(x)
        if self.config.use_moe:
            x = self.moe(x)
        else:
            x = self.mlp(x)
        x = residual + x
        return x


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


class FinAIModel(nn.Module):
    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FinAIBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = FinAIRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask, position_ids)
        x = self.norm(x)
        return x


class FinAIForCausalLM(FinAIPreTrainedModel, GenerationMixin):
    def __init__(self, config: FinAIConfig):
        super().__init__(config)
        self.model = FinAIModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP Auxiliary Heads (Predict next 3 tokens)
        self.mtp_heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.num_mtp_heads - 1)
            ]
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self, input_ids, labels=None, attention_mask=None, position_ids=None, **kwargs
    ):
        hidden_states = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Main next-token loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

            # MTP losses (Auxiliary next-k-token prediction)
            if self.config.num_mtp_heads > 1:
                # mtp_weight=0.5 split among auxiliary heads
                mtp_weight_per_head = self.config.mtp_weight / (
                    self.config.num_mtp_heads - 1
                )
                for i, head in enumerate(self.mtp_heads):
                    # head 0 predicts t+2, head 1 predicts t+3, head 2 predicts t+4
                    offset = i + 2
                    if labels.shape[1] > offset:
                        mtp_logits = head(hidden_states[..., :-offset, :]).contiguous()
                        mtp_labels = labels[..., offset:].contiguous()
                        mtp_loss = F.cross_entropy(
                            mtp_logits.view(-1, self.config.vocab_size),
                            mtp_labels.view(-1),
                        )
                        loss = loss + mtp_weight_per_head * mtp_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
