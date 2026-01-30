# FinAI-Next: Frontier Liquid-BitNet 🚀

FinAI-Next is a state-of-the-art, 331M parameter **Large Language Model (LLM)** built from the ground up for extreme efficiency and financial reasoning. It leverages a revolutionary **Liquid-BitNet** architecture, combining ternary quantization with adaptive dynamical systems.

## 🌟 Key Innovations

### 1. Liquid-BitNet Architecture
Unlike standard Transformers that suffer from the $O(n^2)$ attention bottleneck, FinAI-Next uses **Liquid Dynamical Blocks**.
- **Linear Scaling**: Context length scales linearly ($O(n)$), enabling native **32k+ context windows** on consumer CPUs.
- **Adaptive State Evolution**: The model dynamically adjusts its internal "hidden state" velocity based on the complexity of the input tokens.

### 2. Ternary Quantization (BitNet b1.58)
FinAI-Next uses native ternary weights $\{-1, 0, 1\}$.
- **Energy Efficient**: Replaces expensive floating-point multiplications with simple integer additions/subtractions.
- **CPU Friendly**: Designed to run at high speeds on standard laptop/desktop CPUs without requiring high-end GPUs.

### 3. Adaptive Compute & Multimodal
- **Dynamic Depth**: Skips layers when processing simple text, saving up to 40% of compute during inference.
- **Multimodal Ready**: Built-in projectors for **Vision** and **Audio** integration.

## 🛠 Project Structure

- `fin_ai/model/`: Core architecture (BitNet, LiquidBlocks, Adaptive Compute).
- `fin_ai/training/`: Specialized `TernaryTrainer` for high-precision master weight management.
- `train.py`: Main entry point with automated state persistence and checkpointing.
- `.github/workflows/`: Automated **Hourly Training** pipeline.

## 🚀 Hourly Training Pipeline
FinAI-Next is designed for continuous evolution. Every hour, a GitHub Action:
1. **Pulls** the latest weights from Hugging Face.
2. **Trains** on a fresh slice of the `fineweb-edu` dataset.
3. **Pushes** the updated weights back to Hugging Face.
4. **Persists** the dataset progress (using `dataset_state.json`).

## 📈 Monitoring
Real-time training metrics (Loss, Learning Rate, Token Throughput) are tracked via **Comet ML**.
[View Live Dashboard](https://www.comet.com/meridianalgo/finai-next)

## 📎 Technical Specifications
- **Parameters**: 331,296,816
- **Hidden Size**: 1536
- **Layers**: 24
- **State Dim**: 384
- **Vocabulary**: 151,665 (Qwen2.5 optimized)
- **Quantization**: 1.58-bit (Ternary)

---
*Developed by MeridianAlgo for the next generation of efficient financial intelligence.*
