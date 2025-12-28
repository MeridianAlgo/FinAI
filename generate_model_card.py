#!/usr/bin/env python3
"""Generate an enhanced Hugging Face model card with training stats."""

import json
import os
from datetime import datetime
from pathlib import Path

def load_env():
    """Load .env file."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

def get_training_stats():
    """Get training statistics from checkpoint."""
    version_path = "checkpoints/version.json"
    if os.path.exists(version_path):
        with open(version_path) as f:
            return json.load(f)
    return {}

def generate_model_card():
    """Generate enhanced model card."""
    stats = get_training_stats()
    
    model_card = f"""---
license: mit
tags:
  - pytorch
  - gpt2
  - text-generation
  - fin-ai
  - experimental
  - in-training
  - from-scratch
  - automated-training
language:
  - en
datasets:
  - wikitext
  - roneneldan/TinyStories
  - openai/gsm8k
  - squad
  - imdb
  - ag_news
  - yelp_review_full
  - cnn_dailymail
  - billsum
  - commonsense_qa
  - hellaswag
  - winogrande
  - boolq
  - race
  - stanfordnlp/coqa
  - allenai/c4
  - Skylion007/openwebtext
  - trivia_qa
  - hotpot_qa
  - microsoft/ms_marco
  - duorc
  - amazon_polarity
  - zeroshot/twitter-financial-news-sentiment
  - sciq
  - quail
  - wiki_qa
  - paws
  - medical_questions_pairs
  - app_reviews
  - rotten_tomatoes
metrics:
  - perplexity
library_name: pytorch
pipeline_tag: text-generation
---

# 🤖 Fin.AI v2.0 - Continuously Trained Language Model

<div align="center">

![Status](https://img.shields.io/badge/status-training-yellow)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Parameters](https://img.shields.io/badge/parameters-30M-green)
![License](https://img.shields.io/badge/license-MIT-blue)

**⚠️ EXPERIMENTAL MODEL - Training from scratch**

[GitHub](https://github.com/MeridianAlgo/FinAI) • [Training Logs](https://wandb.ai/meridianalgo-meridianalgo/fin-ai) • [Report Issue](https://github.com/MeridianAlgo/FinAI/issues)

</div>

---

## 🚨 Important Notice

**This model is training from scratch and outputs will be gibberish initially.**

- 🔴 **Brand new model** - Starting from random weights
- ⏳ **Training time needed**: 2-4 weeks for basic coherence
- 🤖 **Automated training**: Every 1 hour 10 minutes via GitHub Actions
- 📊 **Current quality**: Expect complete nonsense initially
- 🎯 **Purpose**: Research/experimental continuous learning

---

## 📊 Model Overview

| Specification | Value |
|--------------|-------|
| **Architecture** | GPT-2 style Transformer |
| **Parameters** | 30,142,848 (~30M) |
| **Layers** | 6 |
| **Attention Heads** | 6 |
| **Embedding Dimension** | 384 |
| **Feed-Forward Dimension** | 1,536 |
| **Max Sequence Length** | 512 tokens |
| **Vocabulary Size** | 50,257 (GPT-2 tokenizer) |
| **Position Encoding** | Rotary (RoPE) |
| **Activation** | GELU |

---

## 🎯 Training Details

### Training Schedule
- **Frequency**: Every 1 hour 10 minutes (6 cycles/hour)
- **Steps per cycle**: 800 steps
- **Daily steps**: ~115,200 steps
- **Weekly steps**: ~806,400 steps
- **Batch size**: 8 (effective: 32 with gradient accumulation)
- **Learning rate**: 3e-4 with cosine decay
- **Warmup steps**: 100

### Training Infrastructure
- **Platform**: GitHub Actions (free tier)
- **Hardware**: CPU only
- **Training time**: ~15-20 minutes per cycle
- **Automatic upload**: To Hugging Face after each cycle

### Datasets (30 total, rotating hourly)

The model trains on a diverse set of 30 datasets, cycling through one per hour:

**📚 Knowledge & Reference**
- WikiText-2, OpenWebText, C4

**✍️ Creative Writing**
- TinyStories

**📰 News & Articles**
- CNN/DailyMail, AG News, Billsum

**❓ Question Answering**
- SQuAD, CoQA, TriviaQA, HotpotQA, MS MARCO, WikiQA, Quail

**🧠 Reasoning & Logic**
- GSM8K (Math), Common Sense QA, HellaSwag, WinoGrande, BoolQ

**📖 Reading Comprehension**
- RACE, DuoRC

**💬 Reviews & Sentiment**
- IMDB, Yelp, Amazon Polarity, Rotten Tomatoes, App Reviews

**🔬 Scientific & Medical**
- SciQ, Medical Questions

**💰 Financial**
- Twitter Financial News

**🔄 Paraphrase & Similarity**
- PAWS

---

## 📈 Training Progress

### Current Status
- **Version**: {stats.get('version', 'v2.0.0')}
- **Training started**: December 28, 2024
- **Model type**: {stats.get('model_type', 'fresh_init')}
- **Total parameters**: {stats.get('parameters', 30142848):,}

### Expected Timeline

| Week | Expected Quality | Description |
|------|-----------------|-------------|
| 1 | 🔴 Gibberish | Random weights, no coherence |
| 2 | 🟠 Patterns | Some token patterns emerging |
| 3-4 | 🟡 Basic | Simple word sequences |
| 5-8 | 🟢 Improving | Short coherent phrases |
| 9-12 | 🔵 Decent | Usable for simple tasks |

### Monitoring
- **GitHub Actions**: [View Training Runs](https://github.com/MeridianAlgo/FinAI/actions)
- **Wandb Dashboard**: [View Metrics](https://wandb.ai/meridianalgo-meridianalgo/fin-ai)
- **Model Updates**: This page updates automatically

---

## 💻 Usage

### Installation

```bash
pip install torch transformers huggingface-hub
```

### Download Model

```python
from huggingface_hub import hf_hub_download
import os

# Create directory
os.makedirs("./fin_ai_model", exist_ok=True)

# Download model files
hf_hub_download("MeridianAlgo/Fin.AI", "model.pt", local_dir="./fin_ai_model")
hf_hub_download("MeridianAlgo/Fin.AI", "config.json", local_dir="./fin_ai_model")
```

### Generate Text (Experimental)

```python
from fin_ai.model import FinAIModel
import torch
from transformers import AutoTokenizer

# Load model
model = FinAIModel.from_pretrained("./fin_ai_model")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate text (expect poor quality initially)
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
    )

generated_text = tokenizer.decode(output[0])
print(generated_text)

# Note: Output quality is poor initially and improves over weeks
```

---

## 🔬 Technical Details

### Architecture Improvements (v2.0)

Compared to v1.x:
- ✅ **3x more parameters** (10M → 30M)
- ✅ **Better architecture** (4 layers → 6 layers)
- ✅ **Larger embeddings** (256 → 384 dimensions)
- ✅ **More attention heads** (4 → 6 heads)
- ✅ **Improved training** (600 → 800 steps/cycle)

### Training Configuration

```yaml
model:
  size_preset: "small"
  n_layers: 6
  n_heads: 6
  embed_dim: 384
  ff_dim: 1536
  max_seq_len: 512

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  weight_decay: 0.01
  warmup_steps: 100
  max_steps: 800
```

---

## 📊 Evaluation

### Metrics Tracked
- **Training Loss**: Cross-entropy loss
- **Perplexity**: exp(loss)
- **Tokens/Second**: Training throughput
- **Learning Rate**: Cosine schedule with warmup
- **Gradient Norm**: For stability monitoring

### Benchmarks (Coming Soon)
Once the model reaches basic coherence, we'll evaluate on:
- HellaSwag (common sense)
- LAMBADA (reading comprehension)
- WikiText perplexity
- Custom generation quality tests

---

## ⚠️ Limitations

1. **Early Training**: Model is in very early training stages
2. **Output Quality**: Expect gibberish for several weeks
3. **CPU Training**: Slower than GPU training
4. **Small Model**: 30M parameters is relatively small
5. **Limited Context**: 512 token context window
6. **No Fine-tuning**: Base model only, not instruction-tuned
7. **English Only**: Trained primarily on English text

---

## 🤝 Contributing

This is an open research project! Contributions welcome:

- **Code**: [GitHub Repository](https://github.com/MeridianAlgo/FinAI)
- **Issues**: [Report Problems](https://github.com/MeridianAlgo/FinAI/issues)
- **Discussions**: [Join Discussion](https://github.com/MeridianAlgo/FinAI/discussions)

---

## 📜 License

MIT License - See [LICENSE](https://github.com/MeridianAlgo/FinAI/blob/main/LICENSE)

---

## 📚 Citation

```bibtex
@misc{{finai2024,
  title={{Fin.AI: Continuously Trained Language Model}},
  author={{MeridianAlgo}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/MeridianAlgo/Fin.AI}}}},
  note={{Experimental model in active training}}
}}
```

---

## 🔗 Links

- **Repository**: https://github.com/MeridianAlgo/FinAI
- **Training Logs**: https://wandb.ai/meridianalgo-meridianalgo/fin-ai
- **GitHub Actions**: https://github.com/MeridianAlgo/FinAI/actions
- **Issues**: https://github.com/MeridianAlgo/FinAI/issues

---

<div align="center">

**Last Updated**: {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}

**Status**: 🔴 Training from Scratch

**Quality**: ⚠️ Expect Gibberish (2-4 weeks needed)

</div>
"""
    
    return model_card

def main():
    """Generate and save model card."""
    print("📝 Generating enhanced model card...")
    
    model_card = generate_model_card()
    
    output_path = "checkpoints/model/README.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    print(f"✅ Model card saved to {output_path}")
    print(f"📊 Size: {len(model_card)} characters")

if __name__ == "__main__":
    main()
