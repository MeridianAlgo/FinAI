# FinAI-Next: Frontier Liquid-BitNet Architecture

[![Architecture](https://img.shields.io/badge/Architecture-Liquid--BitNet-blueviolet?style=flat-square)](https://github.com/MeridianAlgo/FinAI)
[![Parameters](https://img.shields.io/badge/Parameters-331M-blue?style=flat-square)](https://huggingface.co/MeridianAlgo/FinAI-Lite)
[![Precision](https://img.shields.io/badge/Precision-1.58--bit-green?style=flat-square)](https://arxiv.org/abs/2402.17764)
[![Context](https://img.shields.io/badge/Context-32k+-orange?style=flat-square)](https://github.com/MeridianAlgo/FinAI)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

FinAI-Next is a professional-grade, 331M parameter Large Language Model (LLM) engineered for high-efficiency financial reasoning and long-context processing. The architecture integrates BitNet b1.58 ternary quantization with Liquid Dynamical Systems, enabling frontier-class performance on standard consumer hardware.

## Core Architectural Innovations

### 1. Liquid-BitNet Sequence Modeling
FinAI-Next utilizes Liquid Dynamical Blocks to address the quadratic complexity of traditional Transformer architectures.
- **Linear Complexity ($O(n)$):** Computational requirements scale linearly with sequence length, facilitating native support for context windows exceeding 32k tokens.
- **Stateful Recurrence:** Adaptive dynamical systems evolve the internal hidden state, preserving long-range dependencies without the memory overhead of a KV-cache.

### 2. Ternary Quantization (BitNet b1.58)
The model employs native ternary weights ({-1, 0, 1}), significantly reducing the computational footprint.
- **Multiplication-Free Inference:** High-precision operations are replaced with efficient additions and subtractions.
- **Hardware Optimization:** Primary design focus on CPU execution, ensuring high-speed inference on standard desktop and mobile processors.

### 3. Adaptive Compute and Multimodal Integration
- **Dynamic Depth:** Implements token-wise confidence gating to skip layers during low-complexity processing, reducing latency by up to 40%.
- **Multimodal Projectors:** Unified architectural support for Vision and Audio feature mapping into the core latent space.

## Project Structure

- `fin_ai/model/`: Neural engine implementation including BitNet and LiquidBlock modules.
- `fin_ai/training/`: Specialized TernaryTrainer for managing high-precision master weights and low-bit gradients.
- `train.py`: Primary training interface with integrated state persistence and checkpointing.
- `.github/workflows/`: Continuous evolution pipeline for scheduled hourly training.

## Automated Training Pipeline
FinAI-Next is designed for continuous parameter evolution. The automated GitHub Actions pipeline performs the following tasks every hour:
1. Retrieval of the most recent model weights from the Hugging Face repository.
2. Iterative training on a specific slice of the Fineweb-Edu dataset.
3. Automated synchronization of updated weights back to Hugging Face.
4. Persistence of dataset iteration state in `dataset_state.json`.

## Technical Specifications
- **Parameters:** 331,296,816 (Effective)
- **Hidden Dimensions:** 1536
- **Network Depth:** 24 Layers
- **State Dimension:** 384
- **Vocabulary:** 151,665 (Qwen2.5 optimized)
- **Precision:** 1.58-bit (Ternary Weights)

## Performance Monitoring
Comprehensive training metrics, including loss convergence and learning rate schedules, are tracked via Comet ML.
[Access the Monitoring Dashboard](https://www.comet.com/meridianalgo/finai-next)

---
*Developed by MeridianAlgo for advanced, efficient financial intelligence.*
