# Meridian.AI Architecture

## Overview

Meridian.AI combines two components:

1. **Training backbone**: `Qwen/Qwen2.5-0.5B` — a pretrained 0.5B-parameter Qwen2 model that is continuously fine-tuned via the hourly CI pipeline.

2. **Custom research module** (`meridian/`): A from-scratch Sparse MoE Transformer (`MeridianForCausalLM`) used in smoke tests, architecture experiments, and as a reference implementation for the design choices applied to the Qwen backbone.

---

## Specifications

| Component | Value |
|:---|:---|
| Training base | Qwen2.5-0.5B |
| Custom arch type | Sparse Mixture-of-Experts Transformer |
| Layers | 14 (alternating Dense ↔ MoE, `moe_layer_frequency=2`) |
| Attention | Grouped Query Attention (GQA) — 12 Q heads, 4 KV heads |
| Head dimension | 64 (hidden_size / num_attention_heads = 768 / 12) |
| KV groups | 3 Q heads per KV head |
| Position encoding | Rotary Position Embeddings (RoPE, theta=500,000) |
| Feed-forward | SwiGLU |
| Normalization | RMSNorm (eps=1e-6) |
| Vocabulary | 151,665 tokens (Qwen2.5 tokenizer) |
| Context window | 2,048 tokens |
| MoE experts/layer | 8 total, top-2 active per token |
| MoE expert FFN size | 896 (intermediate_size / 2) |
| Dense FFN size | 1,792 (≈2.3× hidden_size, SwiGLU ratio) |
| Router aux loss | 0.01 × Switch-style load-balancing loss |
| Weight tying | lm_head ↔ embed_tokens (saves ~116M params) |
| Numeracy encoding | 64-dim learned magnitude embeddings |
| Continual learning | Elastic Weight Consolidation (EWC) |

---

## Component Deep Dive

### RMSNorm

Root Mean Square Layer Normalization. Faster than standard LayerNorm because it omits the mean-centering step. Computation is cast to float32 for stability, then cast back.

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) × weight
```

### Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating query/key vectors in complex space. Unlike learned absolute embeddings, RoPE generalizes naturally to lengths beyond training. We use `theta=500,000` (same as Qwen2.5) for extended context support.

Caching: cos/sin tables are computed once per sequence length and cached. On KV-cache decode steps (single-token forward), only the last position's cos/sin slice is applied.

### Grouped Query Attention (GQA)

Standard multi-head attention (MHA) has `num_heads` independent KV heads. GQA shares KV heads across groups of Q heads:

- 12 Q heads, 4 KV heads → each KV head serves 3 Q heads
- KV cache is 3× smaller than MHA
- Attention computation is identical to MHA after KV expansion

### SwiGLU

```
SwiGLU(x) = down_proj(SiLU(gate_proj(x)) × up_proj(x))
```

Gate projection controls information flow multiplicatively. SiLU (Sigmoid Linear Unit) is smoother than ReLU. The two-branch design improves representation capacity per parameter versus standard FFN.

### Sparse Mixture-of-Experts (SMoE)

Every other layer (layers 1, 3, 5, …, 13) replaces the dense SwiGLU FFN with a MoE layer:

```
Expert router: linear(hidden_size → num_experts)
         ↓
Top-2 softmax probabilities selected per token
         ↓
2 experts run; outputs weighted by router probs
         ↓
Load-balancing loss: num_experts × Σ(f_e × P_e)
```

Where:
- `f_e` = fraction of tokens routed to expert `e` (first-choice only)
- `P_e` = mean softmax probability assigned to expert `e`

This auxiliary loss (scaled by `router_aux_loss_coef=0.01`) discourages all tokens collapsing to a single expert.

**Efficiency**: Experts are batched by assignment — each expert runs a single forward pass over all tokens assigned to it. Index mapping uses `torch.searchsorted` (O(log n)) rather than Python dict lookups.

### Financial Numeracy Encoding

A learned 64-dimensional embedding for numeric magnitude, added to the token embedding layer with a scale factor of 0.05. The embedding is indexed by `input_id % 32` — a heuristic bucket that groups token IDs by modulo position.

> **Known Limitation (v6.0.0 target):** The current `input_id % 32` heuristic has no relationship to actual numeric magnitude — token ID 1234 modulo 32 = 18, which has nothing to do with whether token 1234 decodes to "100" or "0.001". A proper implementation would detect digit tokens in the vocabulary and assign magnitude buckets based on the decoded numeric value. This is tracked as a v6 improvement.

This provides some additional representational capacity for tokens in structured numeric positions, but does not yet capture true financial magnitude signals.

### Elastic Weight Consolidation (EWC)

After each training run, EWC estimates which model parameters were most important for previously learned tasks using the diagonal Fisher Information Matrix:

```
Fisher(θ)_i ≈ E[( ∂ log p(y|x,θ) / ∂θ_i )²]
```

Estimated by averaging squared gradients over `EWC_SAMPLES` batches.

On subsequent runs, a penalty is added to the loss:

```
L_EWC = λ/2 × Σ_i  F_i × (θ_i - θ*_i)²
```

Where `θ*` are the parameter values from the previous run. This penalizes large changes to parameters with high Fisher values — preserving past knowledge.

**Memory optimization**: EWC gradients are applied manually (not via autograd graph) to avoid building a massive computation graph over the full parameter set. Only parameters with `Fisher > threshold` are stored, reducing EWC state size by ~40–60%.

---

## Weight Initialization

- **Linear layers**: Normal distribution, σ = `initializer_range` (default 0.02)
- **Residual-path linear layers** (o_proj, down_proj): σ = `initializer_range / sqrt(2 × num_layers)` — scaled down to prevent gradient explosion at initialization with deep residuals
- **Embeddings**: Normal distribution, σ = `initializer_range`; padding index zeroed

---

## Parameter Count Breakdown

| Module | Parameters |
|:---|:---|
| embed_tokens | 116.5M (151,665 × 768) |
| 7 dense layers × SwiGLU | 7 × (768×1792×3) = 28.9M |
| 7 MoE layers × 8 experts | 7 × 8 × (768×896×3) = 14.5M |
| 14 attention layers | 14 × (768×768×4) ≈ 33.2M |
| 14 × 2 RMSNorm | ~0.02M |
| NumeracyEncoder | ~0.05M |
| lm_head | tied to embed_tokens |
| **Total (with ties)** | **~479M** |
| **Unique (no ties)** | **~283M** |
