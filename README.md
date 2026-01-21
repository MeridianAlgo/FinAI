# Fin.AI

**WORK IN PROGRESS – EXPERIMENTAL RESEARCH PROJECT**

A continuously learning transformer language model that trains automatically every hour on diverse datasets using GitHub Actions.

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

Fin.AI is an experimental GPT-style language model that trains **24/7** with a rotating curriculum of diverse datasets. The model is designed for efficiency on CPU and consumer hardware, using modern transformer architecture with gradient checkpointing and safetensors for safe, fast model distribution.

**Core characteristics:**

- Fully automated hourly training (GitHub Actions)
- 24 diverse dataset categories (news, math, code, dialogue, science, instructions...)
- Focus rotates every hour → targeted capability improvement
- Models automatically pushed to Hugging Face after each run
- Training metrics publicly visible on Comet ML
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
| Rotating Curriculum      | 24 dataset families covering very different capabilities                  |
| Hugging Face Integration | Latest checkpoint pushed automatically after every training cycle        |
| Real-time Monitoring     | Full metrics, loss curves and samples on Comet ML                         |
| Flexible Scale           | Easily switch between ~16M and ~124M parameters                            |
| CPU-friendly             | Optimized to train efficiently on standard GitHub Actions runners         |
| Gradient Checkpointing   | Memory-efficient training on consumer hardware                            |
| Safe Serialization       | Uses safetensors for secure, fast model loading                           |

## Training Curriculum (24-cycle daily rotation)

| Category              | Example Datasets                     | Cycle Hours       | Primary Focus                     |
|-----------------------|--------------------------------------|-------------------|-----------------------------------|
| Encyclopedia          | WikiText                             | 0, 6              | Broad world knowledge             |
| Creative Writing      | TinyStories                          | 1, 18             | Storytelling & narrative          |
| News                  | CNN, AG News, CC News                | 2,15,17,20        | Current events & factual style    |
| Math & Reasoning      | GSM8K, CommonsenseQA                 | 3,9,19,23         | Problem solving & logic           |
| Open Web Text         | OpenWebText, C4                      | 4,11              | Diverse internet language         |
| Question Answering    | SQuAD                                | 5,22              | Reading comprehension             |
| Instruction Following | Alpaca, Dolly                        | 7,14,21           | Following user instructions       |
| Reviews & Sentiment   | IMDB, Amazon, Yelp                   | 8,10,16           | Opinion & sentiment analysis      |
| Scientific / Medical  | PubMed                               | 12                | Scientific & medical literature   |
| Conversations         | UltraChat                            | 13                | Natural dialogue                  |

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

# Run training with default settings
python train.py --config config/model_config.yaml --datasets config/datasets.yaml

# Run training with specific size preset
python train.py --config config/model_config.yaml --datasets config/datasets.yaml --size-preset micro --max-steps 1000
```

## Current Project Status

### Training Status

[![Training Workflow](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml)
[![Daily Evaluation](https://github.com/MeridianAlgo/FinAI/actions/workflows/daily-eval.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/daily-eval.yml)
[![Comet ML](https://img.shields.io/badge/Comet_ML-Experiments-blue?logo=comet)](https://www.comet.com/meridianalgo/fin-ai)

- **Latest checkpoint**: [huggingface.co/MeridianAlgo/Fin.AI](https://huggingface.co/MeridianAlgo/Fin.AI)
- **Training pipeline**: [GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions)
- **Live metrics & samples**: [Comet ML](https://www.comet.com/meridianalgo/fin-ai)
- **Current model size**: Micro (~16M parameters)
- **Training frequency**: Every hour
- **Last training run**: See GitHub Actions for latest status

### CI Status

[![CI - Tests and Lint](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)

- **Tests**: Passing ✅
- **Linting**: Passing ✅ (Black, Ruff, isort)
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
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5e-4
  max_steps: 800
  gradient_checkpointing: true  # Auto-enabled on CPU
  use_comet: true
```

### Hardware Requirements

- **Minimum**: 4GB RAM, any modern CPU
- **Recommended**: 8GB+ RAM, multi-core CPU
- **GPU**: Optional but recommended for larger models (small, base presets)
- **Storage**: ~500MB for model files

### Performance Characteristics

- **Training speed**: ~100-200 steps/hour on GitHub Actions CPU runners (micro preset)
- **Inference speed**: ~50-100 tokens/second on modern CPU
- **Memory usage**: ~200MB RAM during inference (micro preset)

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
