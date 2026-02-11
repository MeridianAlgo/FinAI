# 🌊 MeridianFormer: Sparse MoE Financial LLM

[![MeridianFormer Hourly Training](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml)
[![MeridianFormer CI](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: MoE](https://img.shields.io/badge/Architecture-Sparse%20MoE-green.svg)](#architecture)

> **MeridianFormer** is a state-of-the-art, 283M-parameter Sparse Mixture-of-Experts (SMoE) language model engineered specifically for financial intelligence and mathematical reasoning. Optimized for high-efficiency CPU execution and continuous hourly learning.

---

### 🤖 Development Notice
*This codebase and the MeridianFormer architecture were architected and implemented by **Antigravity AI**.*

---

## 🔬 Architectural Innovations

MeridianFormer represents a significant departure from standard dense transformer architectures, utilizing a suite of novel techniques to achieve state-of-the-art performance on commodity hardware.

### 1. Sparse Mixture-of-Experts (SMoE)
Unlike traditional models that activate every parameter for every token, MeridianFormer utilizes a **Sparse 8-Expert MoE** system. With **Top-2 Routing**, only approximately **196M parameters** are active per token. This achieves the knowledge capacity of a larger model with the inference speed of a much smaller one, enabling viable high-speed training on CPU infrastructure.

### 2. Grouped Query Attention (GQA) & RoPE
We implement **Grouped Query Attention** (12 Query heads, 4 Key-Value heads) to significantly reduce the memory footprint of the KV cache. This is paired with **Rotary Position Embeddings (RoPE)** (base θ=500k), allowing for robust long-context understanding and superior relative position awareness.

### 3. Financial Numeracy Encoding
A novel contribution to financial modeling: we inject **magnitude-aware embeddings** into the hidden state. Standard tokenization often loses the quantitative scale of numbers (e.g., treating "100" and "10000" as unrelated semantic tokens). MeridianFormer's numeracy encoder restores this relationship, enabling true quantitative reasoning.

### 4. Elastic Weight Consolidation (EWC)
To support **Hourly Continual Pre-training**, we utilize **Elastic Weight Consolidation**. By computing the diagonal Fisher Information Matrix after each training run, the model identifies and protects parameters critical to previously learned financial knowledge, effectively solving the "catastrophic forgetting" problem inherent in online learning.

---

## 🛠 Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Total Parameters** | 283,121,536 |
| **Active Parameters** | 196,417,408 (~1.44x efficiency) |
| **Layers** | 14 (Alternating Dense/MoE) |
| **Hidden Dimension** | 768 |
| **Attention Heads** | 12 Q-Heads / 4 KV-Heads (GQA) |
| **Experts** | 8 per MoE layer (Top-2 Router) |
| **Context Window** | 2,048 Tokens |
| **Weight Tying** | Tied Embeddings (Embed ↔ LM Head) |

---

## 📈 Training Curriculum

MeridianFormer is trained on a curated mix of high-signal datasets to ensure financial expertise:

*   **FinanceAlpaca (40%)**: Specialized financial instructions, market analysis, and QA.
*   **OpenMathInstruct-2 (30%)**: Advanced mathematical reasoning and problem solving.
*   **FineWeb-Edu (30%)**: High-quality educational content for foundational knowledge.

### Continuous Training Loop
The model undergoes automated training every hour via GitHub Actions:
1.  **State Recovery**: Pulls the latest checkpoint from HuggingFace Hub.
2.  **Streaming Pre-training**: Ingests new data samples via weighted round-robin streaming.
3.  **EWC Optimization**: Trains with a penalty to preserve important financial weights.
4.  **Distribution**: Pushes updated weights back to the HF Hub and syncs dataset telemetry.

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Analyze Architecture
```bash
python scripts/count_params.py
```

### Training
```bash
# Smoke test (Development)
SMOKE_TEST=1 python train.py

# Full training run
python train.py
```

### Inference
```bash
python scripts/test_generation.py
```

---

## 📂 Project Structure

*   `meridian/model/`: Core Sparse MoE architecture and configuration.
*   `meridian/training/`: Custom training engine with EWC support.
*   `meridian/data/`: Finance-focused streaming curriculum pipeline.
*   `scripts/`: Utilities for seeding, nuking, and evaluating the model.
*   `.github/workflows/`: Automated hourly training and CI/CD pipelines.

---

## ⚖️ License
Distributed under the **MIT License**. See `LICENSE` (if available) for more information.
