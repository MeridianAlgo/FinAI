# Fin.AI 🤖

A lightweight, trainable transformer-based language model with automated daily training via GitHub Actions.

## Features

- **Scalable Architecture**: GPT-style transformer, easily adjustable from tiny (10M) to large (350M+) parameters
- **Automated Training**: Daily training on different Hugging Face datasets via GitHub Actions
- **Day-based Dataset Rotation**: Different dataset trains each day (Monday-Sunday)
- **Checkpoint Management**: Automatic model saving and resumption
- **Wandb Integration**: Real-time training metrics and visualization
- **CPU-Optimized**: Runs efficiently on GitHub Actions free tier (Ubuntu CPU)
- **Easy Configuration**: YAML-based model and dataset configuration

## Quick Start

### Local Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --config config/model_config.yaml --datasets config/datasets.yaml

# Generate text
python generate.py --model checkpoints/model --prompt "Once upon a time"
```

### GitHub Actions (Automated)

The model trains automatically every day at 6 AM UTC. Each day uses a different dataset:

- **Monday**: WikiText-2 (encyclopedia text)
- **Tuesday**: TinyStories (short stories)
- **Wednesday**: CNN News (news articles)
- **Thursday**: Dolly (instruction data)
- **Friday**: arXiv (scientific papers)
- **Saturday**: SQuAD (Q&A data)
- **Sunday**: WikiText-103 (large encyclopedia)

Checkpoints are automatically committed back to the main branch.

## Configuration

### Model Sizes

| Size | Parameters | Layers | Heads | Embed Dim | Speed |
|------|-----------|--------|-------|-----------|-------|
| tiny | ~10M | 4 | 4 | 256 | ⚡ Fast |
| small | ~25M | 6 | 6 | 384 | 🚀 Medium |
| medium | ~85M | 12 | 8 | 512 | 🐢 Slow |
| large | ~350M | 24 | 12 | 768 | 🐌 Very Slow |

Edit `config/model_config.yaml` to change model size:

```yaml
model:
  size_preset: "tiny"  # or small, medium, large
```

### Datasets

Edit `config/datasets.yaml` to customize datasets for each day:

```yaml
datasets:
  - name: "wikitext"
    subset: "wikitext-2-raw-v1"
    split: "train"
    text_column: "text"
    day: 1  # Monday
    max_samples: 100000
```

### Training Parameters

Adjust in `config/model_config.yaml`:

```yaml
training:
  batch_size: 4
  learning_rate: 5.0e-4
  max_steps: 500
  warmup_steps: 100
  eval_steps: 100
```

## Project Structure

```
fin-ai/
├── fin_ai/                 # Main package
│   ├── model/             # Transformer architecture
│   │   ├── config.py      # Model configuration
│   │   └── transformer.py # GPT-style model
│   ├── data/              # Dataset loading
│   │   └── dataset.py     # HF dataset utilities
│   └── training/          # Training loop
│       └── trainer.py     # Trainer with checkpointing
├── config/                # Configuration files
│   ├── model_config.yaml  # Model & training config
│   └── datasets.yaml      # Dataset configuration
├── train.py               # Main training script
├── generate.py            # Text generation script
├── requirements.txt       # Python dependencies
└── .github/workflows/     # GitHub Actions
    └── train.yml          # Daily training workflow
```

## Usage

### Training

```bash
# Train with default config
python train.py

# Override max steps
python train.py --max-steps 1000

# Limit dataset samples (for testing)
python train.py --max-samples 10000

# Custom output directory
python train.py --output-dir ./my_checkpoints
```

### Generation

```bash
# Generate from prompt
python generate.py --prompt "The future of AI"

# Customize generation
python generate.py \
  --model checkpoints/model \
  --prompt "Hello world" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9
```

## Monitoring Training

### Wandb Dashboard

If you have a Wandb account, add your API key as a GitHub secret:

1. Get your API key from [wandb.ai](https://wandb.ai)
2. Add `WANDB_API_KEY` to GitHub repo secrets
3. View live training at [wandb.ai/your-username/fin-ai](https://wandb.ai)

### Local Checkpoints

Checkpoints are saved to `checkpoints/`:

```
checkpoints/
├── model/                 # Latest model
│   ├── config.json
│   └── model.pt
├── checkpoint-100.pt      # Intermediate checkpoints
├── checkpoint-200.pt
└── best_model.pt          # Best evaluation checkpoint
```

## Performance

On GitHub Actions free tier (Ubuntu CPU):

- **Tiny model**: ~16 seconds per step
- **500 steps**: ~2.2 hours (fits in 3-hour limit)
- **Daily training**: ~500 steps per day
- **Monthly**: ~15,000 steps (~7.5M tokens)

## Architecture

Fin.AI uses a GPT-2 style transformer with:

- Multi-head self-attention with rotary positional embeddings
- Feed-forward layers with SwiGLU activation
- Pre-norm architecture for stable training
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (when GPU available)

## Customization

### Add New Datasets

Edit `config/datasets.yaml`:

```yaml
datasets:
  - name: "your-dataset"
    subset: null
    split: "train"
    text_column: "text"
    day: 1
    max_samples: 50000
```

### Change Training Schedule

Edit `.github/workflows/train.yml`:

```yaml
schedule:
  - cron: '0 6 * * *'  # Daily at 6 AM UTC
```

### Adjust Model Size

Edit `config/model_config.yaml`:

```yaml
model:
  size_preset: "small"  # Larger model
```

## Troubleshooting

### Training too slow

- Reduce `batch_size` in config
- Use smaller `size_preset` (tiny)
- Reduce `max_seq_len` to 256

### Out of memory

- Reduce `batch_size`
- Reduce `max_seq_len`
- Use `gradient_accumulation_steps` to simulate larger batches

### Dataset loading fails

- Check dataset name on [Hugging Face](https://huggingface.co/datasets)
- Verify `text_column` matches dataset schema
- Try with `max_samples` limit first

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for enhancement:

- [ ] GPU support for faster training
- [ ] Distributed training across multiple machines
- [ ] Model quantization for inference
- [ ] Web UI for generation
- [ ] Fine-tuning on custom data

## Security

For security concerns, please see [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- Built with [PyTorch](https://pytorch.org)
- Models from [Hugging Face Transformers](https://huggingface.co/transformers)
- Datasets from [Hugging Face Datasets](https://huggingface.co/datasets)
- Monitoring with [Weights & Biases](https://wandb.ai)

## Status

🚀 **Active Development** - Daily training on GitHub Actions

Latest training: Check [GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions)

---

**Questions?** Open an issue on GitHub!
