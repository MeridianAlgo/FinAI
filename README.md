# FinAI-Next: Liquid-BitNet Architecture

[![Architecture](https://img.shields.io/badge/Architecture-Liquid--BitNet-blueviolet?style=flat-square)](https://github.com/MeridianAlgo/FinAI)
[![Parameters](https://img.shields.io/badge/Parameters-518M-blue?style=flat-square)](https://huggingface.co/MeridianAlgo/FinAI-Lite)
[![Precision](https://img.shields.io/badge/Precision-1.58--bit-green?style=flat-square)](https://arxiv.org/abs/2402.17764)
[![Context](https://img.shields.io/badge/Context-32k+-orange?style=flat-square)](https://github.com/MeridianAlgo/FinAI)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

FinAI-Next is a professional-grade, 518M parameter Large Language Model engineered for high-efficiency financial reasoning and long-context processing. The architecture integrates BitNet b1.58 ternary quantization with Liquid Neural Networks, enabling frontier-class performance on standard consumer hardware.

## Core Architectural Innovations

### 1. Liquid-BitNet Sequence Modeling
FinAI-Next utilizes Liquid Neural Network blocks to address the quadratic complexity of traditional Transformer architectures.
- **Linear Complexity (O(n)):** Computational requirements scale linearly with sequence length, facilitating native support for context windows exceeding 32k tokens.
- **Stateful Recurrence:** Adaptive dynamical systems evolve the internal hidden state, preserving long-range dependencies without the memory overhead of a KV-cache.

### 2. Ternary Quantization (BitNet b1.58)
The model employs native ternary weights ({-1, 0, 1}), significantly reducing the computational footprint.
- **Multiplication-Free Inference:** High-precision operations are replaced with efficient additions and subtractions.
- **Hardware Optimization:** Primary design focus on CPU execution, ensuring high-speed inference on standard desktop and mobile processors.

### 3. Adaptive Compute and Multimodal Integration
- **Dynamic Depth:** Implements token-wise confidence gating to skip layers during low-complexity processing, reducing latency by up to 40%.
- **Multimodal Projectors:** Unified architectural support for Vision and Audio feature mapping into the core latent space.

## Project Structure

```
fin_ai/
├── model/          # Neural engine implementation including BitNet and LiquidBlock modules
│   ├── configuration_next.py
│   └── modeling_next.py
├── training/       # Specialized TernaryTrainer for progressive training
│   └── next_trainer.py
├── utils/          # Utility functions
└── __init__.py

train.py            # Primary training interface with integrated state persistence
generate.py         # Text generation script
checkpoint/         # Training checkpoints and model weights
scripts/            # Utility scripts for model management
tests/              # Unit tests
```

## Training Pipeline

The training system supports progressive checkpointing with decreasing loss:

1. **Initial Training:** Model starts with higher loss (e.g., 111.924)
2. **Progressive Runs:** Each subsequent run loads from the previous checkpoint
3. **Loss Convergence:** Loss decreases progressively (e.g., 111.924 -> 10 -> 9 -> ...)
4. **State Persistence:** Dataset position, optimizer state, and scheduler state are saved

### Training Configuration

```python
# Configuration in train.py
config = FinAINextConfig(
    vocab_size=151665,
    hidden_size=1536,
    num_layers=24,
    liquid_state_dim=384,
    gradient_checkpointing=True,
    tie_word_embeddings=False,  # Set to false for checkpoint compatibility
)
```

### Running Training

```bash
# Set environment variables
export MAX_STEPS=200        # Steps per training run
export TOTAL_STEPS=100000    # Total training lifecycle

# Run training
python train.py
```

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Parameters | 518,137,368 |
| Hidden Dimensions | 1536 |
| Network Depth | 24 Layers |
| State Dimension | 384 |
| Vocabulary | 151,665 (Qwen2.5 optimized) |
| Precision | 1.58-bit (Ternary Weights) |
| Max Context | 32,768 tokens |

## Performance Monitoring

Comprehensive training metrics, including loss convergence and learning rate schedules, are tracked via Comet ML.

[Access the Monitoring Dashboard](https://www.comet.com/meridianalgo/finai-next)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `COMET_API_KEY` | API key for Comet ML tracking |
| `HF_TOKEN` | Hugging Face authentication token |
| `MAX_STEPS` | Steps per training run |
| `TOTAL_STEPS` | Total steps across all runs |

## Dependencies

Key dependencies are listed in `requirements.txt`:
- torch
- transformers
- datasets
- accelerate
- comet_ml
- safetensors
- tqdm

## Development Notes

Code formatted via AI-assisted development workflows for consistency and maintainability.

---

*Developed by MeridianAlgo for advanced, efficient financial intelligence.*
