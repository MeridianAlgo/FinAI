# MeridianFormer: Sparse Mixture-of-Experts Financial Language Model

[![MeridianFormer Hourly Training](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml)
[![MeridianFormer CI](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

MeridianFormer is a state-of-the-art 283M-parameter transformer-based language model utilizing a Sparse Mixture-of-Experts (SMoE) architecture. The project is specifically engineered for financial intelligence, high-precision quantitative reasoning, and algorithmic math tasks. It introduces a innovative training paradigm optimized for continuous, hourly execution on standard CPU runners through the application of Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.

---

## Development Attribution
This project is developed by Ishaan M. . All source code has been formatted and edited with the assistance of Antigravity AI. Yes this is my like 50th implementation of this project, but we're pushing for architectural excellence with this one.

---

## Architecture and Technical Implementation

MeridianFormer departs from standard dense architectures to maximize knowledge capacity while minimizing computational overhead during inference and training.

### 1. Sparse Mixture-of-Experts (SMoE)
The model employs a sparse gateway system consisting of 8 distinct experts per MoE layer. Utilizing a load-balanced Top-2 routing mechanism, only 196M parameters are active per token. This architectural choice provides a 1.44x increase in computational efficiency over dense models of equivalent capacity. The router uses a noisy top-k mechanism to ensure expert diversity and is regularized by an auxiliary load-balancing loss to prevent expert collapse.

### 2. Grouped Query Attention (GQA)
To optimize memory utilization and KV-cache efficiency, MeridianFormer implements Grouped Query Attention. With a ratio of 12 Query heads to 4 Key-Value heads, the model significantly reduces the memory bandwidth required for long-context generation. This allows for higher throughput on CPU-bound environments where memory bandwidth is often the primary bottleneck.

### 3. Rotary Position Embeddings (RoPE)
The project utilizes Rotary Position Embeddings with a base frequency (theta) of 500,000. This configuration enables superior relative position awareness and supports context windows of up to 2,048 tokens. The high theta value is specifically tuned for the high-frequency nature of financial data tokens, ensuring stable embeddings even at the edges of the context window.

### 4. SwiGLU Gated Activation
Following state-of-the-art transformer standards (e.g., Llama 3), MeridianFormer replaces standard GELU activations with SwiGLU. The gated linear unit, combined with the SiLU activation function, provides improved gradient flow and representational capacity. The intermediate size is tuned to 1792 for dense layers and 896 for experts to maintain a balanced parameter distribution.

### 5. RMSNorm and Weight Tying
We utilize Root Mean Square Layer Normalization (RMSNorm) for faster computation and stable training. Additionally, input and output embeddings are tied (shared weights), which reduces the total parameter count by ~116M while maintaining superior performance in causal language modeling tasks.

---

## Novel Contributions

### Financial Numeracy Encoding
Standard tokenization strategies often fail to capture the quantitative magnitude of numeric data. MeridianFormer introduces a learned Numeracy Encoding layer that injects magnitude-aware signals into the hidden states. By mapping numeric tokens to their respective logarithmic scales, the model develops an inherent understanding of quantitative relationships (e.g., the order-of-magnitude difference between pricing levels), rather than treating numbers as purely semantic strings.

### Continuous Pre-training via Elastic Weight Consolidation (EWC)
MeridianFormer is designed for perpetual learning. To mitigate the issue of catastrophic forgetting during frequent, incremental training runs, the system implements Elastic Weight Consolidation.
*   **Fisher Information Matrix**: Following each training iteration, the model computes the diagonal Fisher matrix to identify weights that are critical to previously acquired knowledge.
*   **Penalty Mechanism**: During subsequent runs, a L2-style penalty is applied to the loss function for changes made to these critical weights, ensuring the model retains its "core" financial reasoning while adapting to new market data.

---

## System Specifications

| Feature | Specification |
| :--- | :--- |
| **Total Parameters** | 283,121,536 |
| **Active Parameters** | 196,417,408 |
| **Layers** | 14 (Alternating Dense and MoE) |
| **Hidden Dimension** | 768 |
| **Attention Heads** | 12 Query / 4 KV |
| **Expert Count** | 8 (Top-2 Activated) |
| **FFN Intermediate Size** | 1792 (Dense) / 896 (Expert) |
| **Normalization** | RMSNorm (eps=1e-6) |
| **Weight Tying** | Tied Embeddings (Input/Output) |
| **Repo** | MeridianAlgo/FinAI-Lite |

---

## Data Curriculum and Pipeline

The training pipeline uses a weighted streaming curriculum to ensure a balanced foundation in language, mathematics, and finance:

*   **FinanceAlpaca (40%)**: Specialized instruction sets for financial analysis, portfolio management, and market sentiment.
*   **OpenMathInstruct-2 (30%)**: Advanced math reasoning to sharpen the model's logic and calculation capabilities.
*   **FineWeb-Edu (30%)**: High-quality educational data to maintain general semantic proficiency and world knowledge.

Data is streamed via a round-robin generator with support for state persistence (`dataset_state.json`), ensuring that the model never trains on the same sample twice across hourly runs.

---

## Project Execution and Automation

The repository includes a comprehensive GitHub Actions ecosystem (`.github/workflows/`) that manages the model's lifecycle:
*   **Hourly Training**: Automated execution on standard runners, managing checkpoint synchronization with HuggingFace Hub repository `MeridianAlgo/FinAI-Lite`.
*   **HuggingFace Integration**: Seamless weight management using the `safetensors` format for secure and efficient distribution.
*   **Validation**: Integrated CI/CD suite performing architecture smoke tests, parameter validation, and unit testing on every commit.

---

## Developer Guide

### Environment Setup
Dependencies are managed via `requirements.txt`.
```bash
python -m pip install -r requirements.txt
```

### Parameter Analysis
Use the included script to verify the MoE efficiency and parameter distribution.
```bash
python scripts/count_params.py
```

### Training Methods
The `train.py` script supports environment variables for configuration.
```bash
# Smoke test (validate architecture)
SMOKE_TEST=1 python train.py

# Full training iteration
python train.py
```

### Generation and Testing
```bash
python scripts/test_generation.py
```

---

## License
This project is licensed under the MIT License.
