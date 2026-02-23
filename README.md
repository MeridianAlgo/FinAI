# MeridianAI: Financial Intelligence MoE Model

[![MeridianAI Hourly Training](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml)
[![MeridianAI CI](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

MeridianAI is a high-performance financial language model based on the **OpenMoE-650M** architecture. The project is specifically engineered for financial intelligence, high-precision quantitative reasoning, and algorithmic math tasks. It introduces an innovative training paradigm optimized for continuous, hourly execution on standard CPU runners through the application of **Elastic Weight Consolidation (EWC)** to prevent catastrophic forgetting.

---

## Architecture and Technical Foundations

MeridianAI leverages a Sparse Mixture-of-Experts (SMoE) architecture to maximize knowledge capacity while maintaining extreme efficiency during inference and training.

### 1. Sparse Mixture-of-Experts (SMoE)
The model is based on the OpenMoE framework, utilizing a sparse gateway system with 16 distinct experts. By activating only a small subset of parameters per token, the model achieves the representational capacity of a much larger dense model without the associated computational cost. This makes it ideal for deployment on standard CPU environments.

### 2. Elastic Weight Consolidation (EWC)
To support perpetual hourly learning, the model employs Elastic Weight Consolidation. This technique computes the Fisher Information Matrix to identify weights critical to previously learned financial knowledge. During incremental training, a penalty is applied to changes in these weights, ensuring the model retains its core reasoning capabilities while adapting to new market data.

### 3. Quantitative Reasoning & Numeracy
Unlike general-purpose models, MeridianAI is fine-tuned on a specialized curriculum of financial instruction sets and mathematical reasoning data. This ensures high precision when handling quantitative data, algorithmic trading logic, and complex financial analysis.

---

## Model Specifications

| Feature | Specification |
| :--- | :--- |
| **Base Model** | OpenMoE-650M (Sparse MoE) |
| **Total Parameters** | ~830M |
| **Active Parameters** | ~100M-200M per token |
| **Training Method** | Continual Learning with EWC |
| **Domain** | Financial Intelligence / Algorithmic Math |
| **Execution** | CPU-Optimized Hourly Cycles |

---

## Data Pipeline

The training pipeline uses a weighted streaming curriculum to maintain a balanced foundation:
## Model Access
The latest checkpoints are available on the Hugging Face Hub:
[MeridianAlgo/MeridianAI](https://huggingface.co/MeridianAlgo/MeridianAI)

---

## Automation and Deployment

The repository features a fully automated lifecycle via GitHub Actions:
*   **Hourly Continual Learning**: Automated training runs on standard runners.
*   **HuggingFace Integration**: Seamless checkpoint synchronization with the Hub.
*   **EWC Persistence**: Manages and preserves Fisher Information state across runs.

---

## Getting Started

### Environment Setup
```bash
python -m pip install -r requirements.txt
```

### Training
The `train.py` script manages the continual learning loop.
```bash
python train.py
```

---

## License
This project is licensed under the MIT License.

made with love by meridianalgo
