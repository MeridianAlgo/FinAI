# FinAI-Core v2.2 Ultra-Lite

**WORK IN PROGRESS – EXPERIMENTAL RESEARCH PROJECT**

A continuously learning hybrid Mamba-2 + Transformer model that trains automatically on FineWeb-Edu slices using GitHub Actions.

> **Important Notice**  
> FinAI is an **experimental research prototype**.  
> The model is under continuous training and may produce inaccurate, inappropriate, biased, or nonsensical outputs.  
> **Do NOT use for production applications.**  
> Use at your own risk.

<div align="center">

[![Model on Hugging Face](https://img.shields.io/badge/Model-Fin.AI-yellow)](https://huggingface.co/MeridianAlgo/FinAI-Core-v2.2-UltraLite)
[![CI - Tests and Lint](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)
[![Training Workflow](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Overview

FinAI-Core v2.2 Ultra-Lite is an experimental language model featuring a novel hybrid architecture designed for efficiency and performance. It trains continuously on slices of the FineWeb-Edu dataset, leveraging a "Pull-Train-Push" workflow via GitHub Actions.

## Model Architecture: FinAI-Core v2.2

This model implements a cutting-edge hybrid architecture:

*   **Hybrid Core**: Combines **Mamba-2 State Space Models (SSM)** with **Transformer** layers (Mamba Ratio: 0.6).
*   **Mixture of Experts (MoE)**: DeepSeek-style MoE with 6 experts (2 active per token) for efficient scaling.
*   **Multi-Head Latent Attention (MLA)**: Optimized attention mechanism for better long-context performance.
*   **Multi-Token Prediction (MTP)**: Predicts multiple future tokens simultaneously for faster convergence and inference.
*   **Optimization**: Rotary Embeddings (RoPE), RMSNorm, SwiGLU activations.

### Specifications

| Component | Specification |
|-----------|--------------|
| **Hidden Size** | 1280 |
| **Layers** | 20 |
| **Attention Heads** | 10 |
| **MoE Experts** | 6 (2 active) |
| **Mamba Ratio** | 0.6 |
| **Context Length** | 4096 |
| **Vocab Size** | ~51k (GPT-2 + Finance Tokens) |

## Training Configuration

The model trains in a continuous loop:

1.  **Pull**: GitHub Actions runner pulls the latest model checkpoint from Hugging Face.
2.  **Ingest**: Loads a fresh "slice" of the FineWeb-Edu dataset (streaming mode).
3.  **Train**: Trains for ~400 steps on the new slice.
4.  **Push**: Pushes the updated model back to Hugging Face.

### Dataset: FineWeb-Edu

We use the **HuggingFaceFW/fineweb-edu** dataset, a high-quality collection of educational web content. The training process cycles through the dataset to ensuring steady exposure to new tokens.

## Quick Start

### Installation

```bash
pip install torch transformers huggingface_hub
```

### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    "MeridianAlgo/FinAI-Core-v2.2-UltraLite",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate
prompt = "The future of algorithmic trading is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))
```

## Local Training

```bash
# Clone
git clone https://github.com/MeridianAlgo/FinAI.git
cd FinAI

# Install
pip install -r requirements.txt

# Train (requires HF_TOKEN if pushing)
python train.py
```

## License

MIT License