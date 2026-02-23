# Meridian.AI: Financial Intelligence MoE Model

[![Meridian.AI Train](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml)
[![Meridian.AI Lint](https://github.com/MeridianAlgo/FinAI/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Meridian.AI** is a high-performance financial language model utilizing a **Grouped Query Attention (GQA)** backed **Sparse Mixture-of-Experts (SMoE)** architecture. Based on the OpenMoE-650M foundation, the model is specifically engineered for financial intelligence, high-precision quantitative reasoning, algorithmic math tasks, and capital markets analytics.

It introduces an innovative training paradigm optimized for continuous, hourly execution on standard CPU runners, enabled through the application of **Elastic Weight Consolidation (EWC)** to prevent catastrophic forgetting.

---

## Model Access & Hugging Face Deployment

The latest live training checkpoints and full model weights are perpetually synchronized and stored on the Hugging Face Hub. 

 **Hugging Face Repository**: [https://huggingface.co/MeridianAlgo/MeridianAI](https://huggingface.co/MeridianAlgo/MeridianAI)

You can download the model instantly using the `huggingface_hub` Python toolkit or directly through standard `transformers` pipelines.

---

##  Architecture and Technical Foundations

Meridian.AI leverages a custom Sparse Mixture-of-Experts architecture to maximize knowledge capacity while maintaining extreme efficiency during inference and training loops.

### 1. Sparse Mixture-of-Experts (SMoE)
The model functions on a highly sparse active parameter plane. By utilizing a sparse gateway system with distinct experts and activating only a small subset of parameters per token, the model achieves the representational capacity of a much larger dense network without the associated computational cost. This makes it ideal for rapid deployment on standard CPU environments.

### 2. Elastic Weight Consolidation (EWC)
To support perpetual hourly learning natively inside GitHub Actions, the model utilizes EWC (Elastic Weight Consolidation). This technique calculates the Fisher Information Matrix to identify weights critical to previously learned financial knowledge. During incremental training, an active penalty restricts changes in these vital weights, ensuring the model retains its core financial reasoning capabilities while safely adapting to new real-time market data.

### 3. Quantitative Reasoning & Numeracy Encoders
Unlike generic general-purpose models, Meridian.AI boasts embedded numeracy encoders mapping magnitude signals directly into the dense representation. It is continuously fine-tuned on a specialized curriculum of financial instruction sets, real-time news data, and mathematical reasoning arrays to ensure immense precision when handling quantitative logic and complex financial analysis.

---

## Model Specifications

| Feature | Specification |
| :--- | :--- |
| **Model Name** | Meridian.AI |
| **Base Architecture** | OpenMoE-650M (Sparse MoE) |
| **Total Parameters** | ~830M |
| **Active Parameters** | ~100M-200M per token |
| **Training Paradigm** | Hourly Continual Learning + EWC |
| **Domain** | Finance, Algorithmic Trading, Math |
| **Execution** | CPU-Optimized Pipeline |

---

##  Automated Lifecycle & Deployment

The repository possesses a fully autonomous lifecycle governed by GitHub Actions:
*   **Hourly Continual Learning**: Automated pipeline spins up every hour, pulls the dataset, and trains the model continuously on CPU workflows without interruption.
*   **HuggingFace Integration**: Uninterrupted checkpoint synchronization pushing updated `safetensors` to the Hub post-training.
*   **Persistent State Management**: Manages and preserves Fisher Information state mappings and Comet ML iterations across stateless runner instances.

---

##  Getting Started

### Environment Setup
```bash
# Clone the repository and install the custom dependencies
python -m pip install -r requirements.txt
```

### Continual Training Loop
The `train.py` script manages the primary continual learning loop. Set your environments and execute the automated hourly runner format:
```bash
python train.py
```

---

##  License
This project is completely open source and distributed under the **MIT License**.

made with love by meridianalgo 🩵
