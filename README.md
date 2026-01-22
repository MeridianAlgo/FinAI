# Fin.AI

**WORK IN PROGRESS – EXPERIMENTAL RESEARCH PROJECT**

A continuously learning transformer language model that trains automatically every hour on FineWeb-Edu using GitHub Actions.

> **Important Notice**  
> Fin.AI is an **experimental research prototype** and **work in progress**.  
> The model is under continuous training and may produce inaccurate, inappropriate, biased, or nonsensical outputs.  
> **Do NOT use for production applications, critical systems, or high-stakes decisions.**  
> Use at your own risk.

<div align="center">

[![Model on Hugging Face](https://img.shields.io/badge/🤗_Model-Fin.AI-yellow)](https://huggingface.co/MeridianAlgo/Fin.AI)
[![CI - Tests and Lint](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)
[![Training Workflow](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions)
[![Comet ML](https://img.shields.io/badge/Comet_ML-Experiments-blue?logo=comet)](https://www.comet.com/meridianalgo/fin-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Overview

Fin.AI is an experimental GPT-style language model that trains **24/7** continuously on FineWeb-Edu, a high-quality educational web content dataset. The model is designed for efficiency on CPU and consumer hardware, using modern transformer architecture with gradient checkpointing and safetensors for safe, fast model distribution.

**Core characteristics:**

- Fully automated hourly training (GitHub Actions)
- Continuous training on FineWeb-Edu (high-quality educational content)
- 1000 training steps per hour → steady, consistent improvement
- Models automatically pushed to Hugging Face after each run
- Training metrics and loss curves publicly visible on Comet ML
- CPU-optimized with gradient checkpointing for memory efficiency
- Safe serialization using safetensors format

> This is **not** a production-ready model. Expect evolving (and sometimes unstable) behavior.

## Model Architecture (V3)

Fin.AI V3 features a modern transformer architecture optimized for CPU/consumer hardware:

- **Architecture**: GPT-style decoder-only transformer
- **Attention**: Grouped Query Attention (GQA) with Flash Attention support
- **Position Encoding**: Rotary Position Embeddings (RoPE)
- **Activation**: SwiGLU
- **Normalization**: RMSNorm
- **Framework**: Built on HuggingFace Transformers
- **Memory Optimization**: Gradient checkpointing enabled by default on CPU
- **Safe Serialization**: Uses safetensors for secure, efficient model storage

### Model Sizes (Size Presets)

| Preset | Parameters | Layers | Heads | KV Heads | Hidden Dim | FF Dim | Recommended Use Case |
|--------|------------|--------|-------|----------|------------|--------|----------------------|
| micro  | ~16M       | 4      | 4     | 2        | 256        | 1024   | Very fast experiments, CI training |
| small  | ~48M       | 8      | 8     | 4        | 512        | 1792   | Default – good CPU performance |
| base   | ~124M      | 12     | 12    | 6        | 768        | 3072   | Higher quality (GPU recommended) |

**Current deployment**: Micro (16M parameters) - optimized for GitHub Actions CPU runners

## Key Features

| Feature                  | Description                                                                |
|--------------------------|----------------------------------------------------------------------------|
| Automated Continuous Training | Trains every hour – completely hands-free                                 |
| FineWeb-Edu Dataset      | High-quality educational web content for consistent, focused learning     |
| 1000 Steps Per Hour      | Steady progress with ~1000 training steps every hour                      |
| Hugging Face Integration | Latest checkpoint pushed automatically after every training cycle        |
| Real-time Monitoring     | Full metrics, loss curves and samples on Comet ML                         |
| Flexible Scale           | Easily switch between ~16M and ~124M parameters                            |
| CPU-friendly             | Optimized to train efficiently on standard GitHub Actions runners         |
| Gradient Checkpointing   | Memory-efficient training on consumer hardware                            |
| Safe Serialization       | Uses safetensors for secure, fast model loading                           |

## Training Dataset

### FineWeb-Edu

Fin.AI trains continuously on **FineWeb-Edu**, a high-quality educational web content dataset curated by Hugging Face. This dataset provides:

- **High-quality content**: Filtered for educational value and quality
- **Diverse topics**: Covers a wide range of educational subjects
- **Consistent training**: Single dataset allows for steady, predictable improvement
- **Large scale**: 10BT sample provides extensive training material
- **Educational focus**: Content optimized for learning and knowledge acquisition

**Training Schedule:**
- **Frequency**: Every hour, automatically via GitHub Actions
- **Steps per run**: 1000 training steps
- **Checkpointing**: Model state saved every 500 steps
- **Progress tracking**: All metrics and loss curves visible on Comet ML

This continuous training approach allows you to watch the model improve in real-time as loss curves decrease and perplexity improves with each hourly training session.

## Quick Start

### Installation

```bash
pip install transformers torch huggingface_hub
```

### Download Latest Model from Hugging Face

```python
from huggingface_hub import snapshot_download

# Download the entire model directory
model_path = snapshot_download(repo_id="MeridianAlgo/Fin.AI")

# Or download specific files
from huggingface_hub import hf_hub_download
config_path = hf_hub_download("MeridianAlgo/Fin.AI", "config.json")
model_path = hf_hub_download("MeridianAlgo/Fin.AI", "model.safetensors")
```

### Basic Inference Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    "MeridianAlgo/Fin.AI",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Advanced Usage with Custom Generation Config

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "MeridianAlgo/Fin.AI",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Custom generation config
generation_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

prompt = "Explain machine learning in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Local Training

```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/FinAI.git
cd FinAI

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up environment variables for HF sync
# Create .env file (DO NOT COMMIT!)
echo "HF_TOKEN=your_hf_token_here" > .env
echo "COMET_API_KEY=your_comet_key_here" >> .env

# Run training with default settings
python train.py --config config/model_config.yaml --datasets config/datasets.yaml

# Run training with specific size preset
python train.py --config config/model_config.yaml --datasets config/datasets.yaml --size-preset micro --max-steps 1000
```

**Note**: The `.env` file is gitignored and should never be committed. For CI/CD, use GitHub repository secrets instead.

## Current Project Status

### Training Status

[![Training Workflow](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml)
[![Daily Evaluation](https://github.com/MeridianAlgo/FinAI/actions/workflows/daily-eval.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/daily-eval.yml)
[![Comet ML](https://img.shields.io/badge/Comet_ML-Experiments-blue?logo=comet)](https://www.comet.com/meridianalgo/fin-ai)

- **Latest checkpoint**: [huggingface.co/MeridianAlgo/Fin.AI](https://huggingface.co/MeridianAlgo/Fin.AI)
- **Training pipeline**: [GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions)
- **Live metrics & loss curves**: [Comet ML](https://www.comet.com/meridianalgo/fin-ai)
- **Current model size**: Micro (~16M parameters)
- **Training frequency**: Every hour (1000 steps per run)
- **Dataset**: FineWeb-Edu (continuous)
- **Last training run**: See GitHub Actions for latest status

### CI Status

[![CI - Tests and Lint](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)

- **Python versions**: 3.10, 3.11, 3.12

<!-- DAILY_EVAL_START -->
### Daily Model Evolution

Track how the model's responses evolve as it trains continuously!

**Test Prompt:** "The future of artificial intelligence is"

**Latest Responses (Last 7 Days):**

| Date | Response Preview |
|------|------------------|
| *Awaiting first evaluation* | Run the daily-eval workflow to see results |

*The model is evaluated daily with the same prompt to showcase its learning progress.*
<!-- DAILY_EVAL_END -->

## Limitations

- **Experimental**: This is a research project, not production-ready
- **Accuracy**: May produce factual errors or hallucinations
- **Bias**: May reflect biases present in training data
- **Safety**: No safety alignment or RLHF applied
- **Context**: Limited to 1024 tokens (configurable)
- **Scale**: Relatively small (16M parameters in current deployment)
- **Training**: Continuously evolving model with unstable behavior

## Technical Details

### Model Configuration

```yaml
model:
  size_preset: micro  # micro, small, or base
  vocab_size: 50257
  max_seq_len: 1024
  dropout: 0.1
  activation: swiglu
  use_flash_attention: true  # Auto-disabled on CPU
  rope_theta: 10000.0

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 3e-4
  max_steps: 1000  # 1000 steps per hourly run
  gradient_checkpointing: true  # Auto-enabled on CPU
  use_comet: true
```

### Hardware Requirements

- **Minimum**: 4GB RAM, any modern CPU
- **Recommended**: 8GB+ RAM, multi-core CPU
- **GPU**: Optional but recommended for larger models (small, base presets)
- **Storage**: ~500MB for model files

### Performance Characteristics

- **Training speed**: 1000 steps/hour on GitHub Actions CPU runners (micro preset)
- **Inference speed**: ~50-100 tokens/second on modern CPU
- **Memory usage**: ~200MB RAM during inference (micro preset)
- **Dataset**: FineWeb-Edu (streaming, continuous training)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE)


## Links

- **GitHub**: [MeridianAlgo/FinAI](https://github.com/MeridianAlgo/FinAI)
- **Hugging Face**: [MeridianAlgo/Fin.AI](https://huggingface.co/MeridianAlgo/Fin.AI)
- **Training Metrics**: [Comet ML](https://www.comet.com/meridianalgo/fin-ai)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/FinAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/FinAI/discussions)

---

<div align="center">

**Made with passion by the Fin.AI team**  
[⭐ Star on GitHub](https://github.com/MeridianAlgo/FinAI)  [🤗 View & download on Hugging Face](https://huggingface.co/MeridianAlgo/Fin.AI)

</div>
