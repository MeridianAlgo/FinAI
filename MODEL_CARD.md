---
language:
- en
license: mit
tags:
- transformer
- pytorch
- causal-lm
- continuous-training
- gpt
- autoregressive
- text-generation
datasets:
- wikitext
- c4
- squad
- gsm8k
- tinnystories
- alpaca
pipeline_tag: text-generation
library_name: transformers
inference:
  parameters:
    temperature: 0.8
    max_length: 100
    top_p: 0.95
---

# Fin.AI V3

**WORK IN PROGRESS – EXPERIMENTAL RESEARCH PROJECT**

A continuously learning transformer language model that trains automatically every hour on diverse datasets using GitHub Actions.

> **Important Notice**  
> Fin.AI is an **experimental research prototype** and **work in progress**.  
> The model is under continuous training and may produce inaccurate, inappropriate, biased, or nonsensical outputs.  
> **Do NOT use for production applications, critical systems, or high-stakes decisions.**  
> Use at your own risk.

---

## Model Overview

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

---

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

---

## Usage

### Installation

```bash
pip install transformers torch huggingface_hub
```

### Download the Model

#### Option 1: Using Hugging Face Hub (Recommended)

```python
from huggingface_hub import snapshot_download

# Download the entire model directory
model_path = snapshot_download(repo_id="MeridianAlgo/Fin.AI")

# Or download specific files
from huggingface_hub import hf_hub_download
config_path = hf_hub_download("MeridianAlgo/Fin.AI", "config.json")
model_path = hf_hub_download("MeridianAlgo/Fin.AI", "model.safetensors")
```

#### Option 2: Using Transformers (AutoModel)

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
```

### Basic Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "MeridianAlgo/Fin.AI",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

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

### Advanced Usage

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

### Model Configuration

You can inspect the model configuration:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("MeridianAlgo/Fin.AI", trust_remote_code=True)
print(config)
```

---

## Training

### Training Curriculum (24-cycle daily rotation)

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

### Training Configuration

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

---

## Model Status

### Training Status

- **Latest checkpoint**: Available on this Hugging Face repository
- **Training pipeline**: [GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions)
- **Live metrics & samples**: [Comet ML](https://www.comet.com/meridianalgo/fin-ai)
- **Current model size**: Micro (~16M parameters)
- **Training frequency**: Every hour
- **Last training run**: See GitHub Actions for latest status

### CI Status

- **Tests**: Passing ✅
- **Linting**: Passing ✅ (Black, Ruff, isort)
- **Python versions**: 3.10, 3.11, 3.12

---

## Limitations

- **Experimental**: This is a research project, not production-ready
- **Accuracy**: May produce factual errors or hallucinations
- **Bias**: May reflect biases present in training data
- **Safety**: No safety alignment or RLHF applied
- **Context**: Limited to 1024 tokens (configurable)
- **Scale**: Relatively small (16M parameters in current deployment)
- **Training**: Continuously evolving model with unstable behavior

---

## Technical Details

### Model Files

This repository contains the following files:

- `config.json` - Model configuration
- `model.safetensors` - Model weights in safetensors format
- `generation_config.json` - Default generation parameters
- `configuration_finai.py` - Custom configuration class
- `modeling_finai.py` - Custom model architecture

### Safetensors

This model uses the [safetensors](https://github.com/huggingface/safetensors) format for storing model weights. Safetensors is a safe and efficient format that:

- Prevents arbitrary code execution during loading
- Provides faster loading times
- Uses less disk space compared to traditional PyTorch `.bin` files

### Tokenizer

The model uses the GPT-2 tokenizer with a vocabulary size of 50,257 tokens.

---

## Citation

```bibtex
@software{finai2026,
  author = {Fin.AI Team},
  title = {Fin.AI: A Continuously Learning Transformer Language Model},
  year = {2026},
  url = {https://github.com/MeridianAlgo/FinAI},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

MIT License - See [LICENSE](https://github.com/MeridianAlgo/FinAI/blob/main/LICENSE)

---

## Links

- **GitHub**: [MeridianAlgo/FinAI](https://github.com/MeridianAlgo/FinAI)
- **Training Metrics**: [Comet ML](https://www.comet.com/meridianalgo/fin-ai)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/FinAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/FinAI/discussions)

---

**Last Updated**: Auto-updated with each training run
