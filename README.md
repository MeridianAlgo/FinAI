# 🤖 Fin.AI

**A continuously-learning transformer language model trained hourly on diverse datasets via GitHub Actions**

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-Fin.AI-yellow)](https://huggingface.co/MeridianAlgo/Fin.AI)
[![GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## 🌟 What is Fin.AI?

Fin.AI is an experimental GPT-style language model that trains itself **every 1.5 hours** on different datasets from Hugging Face. It's designed to be:

- **🔄 Continuously Learning**: Trains 24/7 on GitHub Actions
- **📚 Diverse**: Rotates through 24 different dataset types
- **🎯 Focused**: Each cycle targets specific capabilities (math, reasoning, conversation, etc.)
- **🚀 Accessible**: Free to use, modify, and deploy
- **📊 Transparent**: All training metrics visible on Wandb

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Automated Training** | Training every 1.5 hours via GitHub Actions (no manual intervention) |
| **Dataset Rotation** | 24 unique datasets covering news, math, code, conversations, and more |
| **Hugging Face Integration** | Model auto-uploaded to [HF Hub](https://huggingface.co/MeridianAlgo/Fin.AI) after each run |
| **Wandb Monitoring** | Real-time training metrics and loss curves |
| **Scalable Architecture** | Easily adjust from 10M to 350M+ parameters |
| **CPU Optimized** | Runs efficiently on free GitHub Actions runners |

## 🎓 Training Curriculum

Fin.AI trains on a diverse curriculum that rotates every 1.5 hours (16 cycles per day):

### Dataset Categories

| Category | Datasets | Hours | Purpose |
|----------|----------|-------|---------|
| 📖 **Encyclopedia** | WikiText | 0, 6 | General knowledge |
| ✍️ **Creative Writing** | TinyStories | 1, 18 | Narrative generation |
| 📰 **News** | CNN, AG News, CC News | 2, 15, 17, 20 | Current events |
| 🧮 **Math & Reasoning** | GSM8K, CommonsenseQA | 3, 9, 19, 23 | Problem solving |
| 🌐 **Web Content** | OpenWebText, C4 | 4, 11 | Internet text |
| ❓ **Q&A** | SQuAD | 5, 22 | Question answering |
| 📋 **Instructions** | Alpaca, Dolly | 7, 14, 21 | Task following |
| ⭐ **Reviews** | IMDB, Amazon, Yelp | 8, 10, 16 | Sentiment analysis |
| 🏥 **Medical** | PubMed | 12 | Scientific text |
| 💬 **Conversations** | UltraChat | 13 | Dialogue |

## 🚀 Quick Start

### Download Pre-trained Model

```python
from huggingface_hub import hf_hub_download

# Download latest model
hf_hub_download("MeridianAlgo/Fin.AI", "model.pt", local_dir="./model")
hf_hub_download("MeridianAlgo/Fin.AI", "config.json", local_dir="./model")
```

### Generate Text

```python
from fin_ai.model import FinAIModel
import torch

# Load model
model = FinAIModel.from_pretrained("./model")
tokenizer = model.tokenizer

# Generate
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.8)
print(tokenizer.decode(outputs[0]))
```

### Train Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train on current hour's dataset
python train.py --config config/model_config.yaml --datasets config/datasets.yaml

# Train with custom settings
python train.py --max-steps 1000 --max-samples 50000
```

## 📊 Model Architecture

Fin.AI uses a modern GPT-2 style transformer with improvements:

- **Multi-head Self-Attention** with rotary positional embeddings (RoPE)
- **SwiGLU Activation** in feed-forward layers
- **Pre-norm Architecture** for training stability
- **Gradient Accumulation** for larger effective batch sizes
- **Mixed Precision Training** (when GPU available)

### Model Sizes

| Preset | Parameters | Layers | Heads | Embed Dim | Use Case |
|--------|-----------|--------|-------|-----------|----------|
| `tiny` | ~10M | 4 | 4 | 256 | Fast prototyping, CPU training |
| `small` | ~25M | 6 | 6 | 384 | Balanced performance |
| `medium` | ~85M | 12 | 8 | 512 | Better quality, slower |
| `large` | ~350M | 24 | 12 | 768 | Best quality, GPU recommended |

Current deployment: **tiny** (optimized for GitHub Actions CPU)

## 📈 Training Performance

On GitHub Actions free tier (Ubuntu CPU):

- **Training Speed**: ~16 seconds/step
- **Hourly Training**: 500 steps (~2 hours)
- **Daily Progress**: ~12,000 steps
- **Monthly Progress**: ~360,000 steps (~180M tokens)

### Expected Metrics

- **Initial Loss**: ~10-15
- **After 500 steps**: ~2-5
- **Learning Rate**: 3e-4 with cosine decay
- **Batch Size**: 8 (effective)

## 🛠️ Configuration

### Change Model Size

Edit `config/model_config.yaml`:

```yaml
model:
  size_preset: "small"  # tiny, small, medium, or large
```

### Customize Training

```yaml
training:
  batch_size: 8
  learning_rate: 3.0e-4
  max_steps: 500
  warmup_steps: 50
```

### Add Custom Datasets

Edit `config/datasets.yaml`:

```yaml
datasets:
  - name: "your-org/your-dataset"
    subset: null
    split: "train"
    text_column: "text"
    max_samples: 20000
```

### Adjust Training Schedule

Edit `.github/workflows/train.yml`:

```yaml
schedule:
  - cron: '0 * * * *'  # Every hour
  # Or customize:
  # - cron: '0 */2 * * *'  # Every 2 hours
  # - cron: '0 0,6,12,18 * * *'  # 4 times daily
```

## 📁 Project Structure

```
fin-ai/
├── fin_ai/                    # Main package
│   ├── model/                 # Transformer implementation
│   │   ├── config.py          # Model configuration
│   │   └── transformer.py     # GPT architecture
│   ├── data/                  # Dataset utilities
│   │   └── dataset.py         # HF dataset loading
│   └── training/              # Training loop
│       └── trainer.py         # Trainer with checkpointing
├── config/                    # Configuration files
│   ├── model_config.yaml      # Model & training settings
│   └── datasets.yaml          # Dataset rotation schedule
├── .github/workflows/         # CI/CD
│   └── train.yml              # Hourly training workflow
├── train.py                   # Training script
├── generate.py                # Text generation
├── test_model.py              # Model tests
└── requirements.txt           # Dependencies
```

## 🔬 Usage Examples

### Basic Generation

```bash
python generate.py --prompt "Once upon a time" --max-tokens 200
```

### Advanced Generation

```bash
python generate.py \
  --model checkpoints/model \
  --prompt "Explain quantum computing" \
  --max-tokens 300 \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.9
```

### Training with Limits

```bash
# Quick test run
python train.py --max-steps 100 --max-samples 5000

# Full training
python train.py --max-steps 1000
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Add more diverse datasets (code, multilingual, etc.)
- [ ] Implement model quantization for faster inference
- [ ] Create web UI for text generation
- [ ] Add evaluation benchmarks
- [ ] Support distributed training
- [ ] Implement LoRA fine-tuning

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔒 Security

For security concerns, see [SECURITY.md](SECURITY.md).

## 📜 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org) - Deep learning framework
- [Hugging Face](https://huggingface.co) - Models and datasets
- [Weights & Biases](https://wandb.ai) - Experiment tracking
- [GitHub Actions](https://github.com/features/actions) - CI/CD

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/FinAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/FinAI/discussions)
- **Model**: [Hugging Face](https://huggingface.co/MeridianAlgo/Fin.AI)

## 📈 Status

🟢 **Active Development** - Training 24/7

- **Latest Model**: [huggingface.co/MeridianAlgo/Fin.AI](https://huggingface.co/MeridianAlgo/Fin.AI)
- **Training Logs**: [GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions)
- **Metrics**: [Wandb Dashboard](https://wandb.ai/meridianalgo-meridianalgo/fin-ai)

---

<div align="center">

**Made with ❤️ by the Fin.AI team**

[⭐ Star us on GitHub](https://github.com/MeridianAlgo/FinAI) • [🤗 Try on Hugging Face](https://huggingface.co/MeridianAlgo/Fin.AI)

</div>
