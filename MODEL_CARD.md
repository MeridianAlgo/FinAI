---
language:
- en
license: mit
tags:
- finai
- mamba
- transformer
- moe
- mtp
- continuous-learning
datasets:
- HuggingFaceFW/fineweb-edu
pipeline_tag: text-generation
inference: false
---

# FinAI-Core v2.2 Ultra-Lite

**FinAI-Core v2.2 Ultra-Lite** is an experimental hybrid language model designed for efficient continuous learning. It combines Mamba-2 State Space Models with Transformer layers, augmented by Mixture of Experts (MoE) and Multi-Token Prediction (MTP).

## Model Details

*   **Developer**: MeridianAlgo (Experimental)
*   **Architecture**: Hybrid Mamba-2 + Transformer (Decoder-only)
*   **Parameter Count**: ~1.2B (Active parameters significantly lower due to MoE)
*   **Context Length**: 4096 tokens
*   **Training Data**: FineWeb-Edu (streaming slices)

### Novel Features

1.  **Hybrid Backbone**: 60% Mamba-2 layers for linear-time sequence modeling, 40% Transformer layers (with MLA) for complex reasoning.
2.  **DeepSeek-style MoE**: 6 experts with 2 active experts per token, balancing knowledge capacity with inference speed.
3.  **Multi-Head Latent Attention (MLA)**: Low-rank key-value projection for efficient attention.
4.  **Multi-Token Prediction (MTP)**: Predicts 3 future tokens at once during training to improve sample efficiency and latent representation.

## Training

The model is trained continuously via GitHub Actions.
*   **Cycle**: Hourly
*   **Data**: Slices of `HuggingFaceFW/fineweb-edu`
*   **Optimizer**: AdamW (8-bit if available)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("MeridianAlgo/FinAI-Lite", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

## Limitations

*   **Experimental**: Not for production use.
*   **Bias**: May reflect biases in web data.
*   **Stability**: Weights change hourly; downstream performance may vary.