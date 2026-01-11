---
language:
- en
license: mit
tags:
- transformer
- pytorch
- causal-lm
- finance
- continuous-training
datasets:
- wikitext
- c4
- squad
- gsm8k
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

A continuously learning transformer language model that trains automatically every ~1.5 hours on diverse datasets using GitHub Actions.

> **Important Notice**  
> Fin.AI is an **experimental research prototype** and **work in progress**.  
> The model is under continuous training and may produce inaccurate, inappropriate, biased, or nonsensical outputs.  
> **Do NOT use for production applications, critical systems, or high-stakes decisions.**  
> Use at your own risk.

## Model Architecture (V3)

This is **Fin.AI V3**, featuring a modern transformer architecture:

- **Architecture**: GPT-style decoder-only transformer
- **Attention**: Grouped Query Attention (GQA) with Flash Attention support
- **Position Encoding**: Rotary Position Embeddings (RoPE)
- **Activation**: SwiGLU
- **Normalization**: RMSNorm
- **Framework**: Built on HuggingFace Transformers

### Model Sizes

| Preset  | Parameters | Layers | Heads | KV Heads | Hidden Dim |
|---------|------------|--------|-------|----------|------------|
| tiny    | ~15M       | 6      | 4     | 2        | 256        |
| small   | ~40M       | 8      | 8     | 4        | 512        |
| medium  | ~120M      | 12     | 12    | 4        | 768        |
| large   | ~350M      | 24     | 16    | 8        | 1024       |

**Current deployment**: Small (40M parameters)

## Training

- **Continuous Training**: Automated training runs every ~1.5 hours
- **Curriculum**: 24 diverse dataset families (news, math, code, dialogue, science, instructions...)
- **Dataset Rotation**: Focus rotates hourly for targeted capability improvement
- **Monitoring**: Full metrics on [Weights & Biases](https://wandb.ai/meridianalgo-meridianalgo/fin-ai)

## Usage

### Installation

```bash
pip install transformers torch
```

### Basic Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("MeridianAlgo/Fin.AI", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Uses GPT-2 tokenizer

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.8,
        top_p=0.95,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Advanced Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("MeridianAlgo/Fin.AI", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Custom generation config
generation_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    do_sample=True
)

prompt = "Explain machine learning in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Curriculum

The model trains on a rotating 24-hour curriculum covering:

- **Encyclopedia**: WikiText
- **Creative Writing**: TinyStories
- **News**: CNN, AG News, CC News
- **Math & Reasoning**: GSM8K, CommonsenseQA
- **Open Web**: OpenWebText, C4
- **Q&A**: SQuAD
- **Instructions**: Alpaca, Dolly
- **Reviews**: IMDB, Amazon, Yelp
- **Scientific**: PubMed
- **Dialogue**: UltraChat

## Limitations

- **Experimental**: This is a research project, not production-ready
- **Accuracy**: May produce factual errors or hallucinations
- **Bias**: May reflect biases present in training data
- **Safety**: No safety alignment or RLHF applied
- **Context**: Limited to 1024 tokens
- **Scale**: Relatively small (40M parameters)

## License

MIT License - See [LICENSE](https://github.com/MeridianAlgo/FinAI/blob/main/LICENSE)

## Links

- **GitHub**: [MeridianAlgo/FinAI](https://github.com/MeridianAlgo/FinAI)
- **Training Metrics**: [Weights & Biases](https://wandb.ai/meridianalgo-meridianalgo/fin-ai)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/FinAI/issues)

## Citation

```bibtex
@software{finai2026,
  author = {Fin.AI Team},
  title = {Fin.AI: A Continuously Learning Transformer Language Model},
  year = {2026},
  url = {https://github.com/MeridianAlgo/FinAI}
}
```

---

**Last Updated**: Auto-updated with each training run
