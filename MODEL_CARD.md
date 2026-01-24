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
- HuggingFaceFW/fineweb-edu
pipeline_tag: text-generation
library_name: transformers
inference:
  parameters:
    temperature: 0.8
    max_length: 100
    top_p: 0.95
---

# Fin.AI

**WORK IN PROGRESS – EXPERIMENTAL RESEARCH PROJECT**

A continuously learning transformer language model that trains automatically every hour on FineWeb-Edu using GitHub Actions.

> **Important Notice**  
> Fin.AI is an **experimental research prototype** and **work in progress**.  
> The model is under continuous training and may produce inaccurate, inappropriate, biased, or nonsensical outputs.  
> **Do NOT use for production applications, critical systems, or high-stakes decisions.**  
> Use at your own risk.

## Model Architecture

Fin.AI features a modern transformer architecture optimized for CPU/consumer hardware:

- **Architecture**: GPT-style decoder-only transformer
- **Attention**: Grouped Query Attention (GQA) with Flash Attention support
- **Position Encoding**: Rotary Position Embeddings (RoPE)
- **Activation**: SwiGLU
- **Normalization**: RMSNorm
- **Framework**: Built on HuggingFace Transformers
- **Memory Optimization**: Gradient checkpointing enabled by default on CPU
- **Safe Serialization**: Uses safetensors for secure, efficient model storage

### Model Specifications

**Current Deployment: Base (124M parameters)**

| Component | Specification |
|-----------|--------------|
| **Total Parameters** | 124,784,896 |
| **Layers** | 12 |
| **Attention Heads** | 12 |
| **KV Heads** | 6 (Grouped Query Attention) |
| **Hidden Dimension** | 768 |
| **Feedforward Dimension** | 3072 |
| **Vocabulary Size** | 50,257 (GPT-2 tokenizer) |
| **Max Sequence Length** | 1024 tokens |
| **Dropout** | 0.1 |
| **RoPE Theta** | 10000.0 |

## Usage

### Installation

```bash
pip install transformers torch huggingface_hub
```

### Basic Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "MeridianAlgo/Fin.AI",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

## Training

### Dataset: FineWeb-Edu

Fin.AI trains continuously on **FineWeb-Edu**, a high-quality educational web content dataset curated by Hugging Face.

**Training Schedule:**
- **Frequency**: Every hour via GitHub Actions
- **Steps per run**: 1000 training steps
- **Checkpointing**: Every 500 steps
- **Timeout**: 85 minutes per run

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 2 |
| **Gradient Accumulation Steps** | 16 |
| **Effective Batch Size** | 32 |
| **Learning Rate** | 3e-4 |
| **Weight Decay** | 0.1 |
| **Warmup Steps** | 1000 |
| **Max Steps per Run** | 1000 |
| **Optimizer** | AdamW |
| **FP16 Training** | Yes (on GPU) |
| **Gradient Checkpointing** | Yes (on CPU) |

## Performance

### Hardware Requirements

- **Minimum**: 4GB RAM, any modern CPU
- **Recommended**: 8GB+ RAM, multi-core CPU
- **GPU**: Optional but recommended for larger models
- **Storage**: ~500MB for model files

### Performance Characteristics

- **Training speed**: ~170-180 tokens/second on GitHub Actions CPU runners
- **Inference speed**: ~50-100 tokens/second on modern CPU
- **Memory usage**: ~200MB RAM during inference (micro preset), ~500MB (base preset)
- **Training time**: ~85 minutes per 1000 steps (base preset on CPU)

## Limitations

- **Experimental**: This is a research project, not production-ready
- **Accuracy**: May produce factual errors or hallucinations
- **Bias**: May reflect biases present in training data
- **Safety**: No safety alignment or RLHF applied
- **Context**: Limited to 1024 tokens (configurable)
- **Scale**: Relatively small (124M parameters in current deployment)
- **Training**: Continuously evolving model with unstable behavior

## Links

- **GitHub**: [MeridianAlgo/FinAI](https://github.com/MeridianAlgo/FinAI)
- **Training Metrics**: [Comet ML](https://www.comet.com/meridianalgo/fin-ai)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/FinAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/FinAI/discussions)

## License

MIT License - See [LICENSE](https://github.com/MeridianAlgo/FinAI/blob/main/LICENSE)

---

**Last Updated**: Auto-updated with each training run
