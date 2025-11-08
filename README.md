# 🚀 FinAI - Financial Language Model

**A modern, production-ready GPT-style language model optimized for financial data and continuous learning.**

FinAI is a lightweight yet powerful transformer-based language model that trains on financial datasets with state-of-the-art optimization techniques. Features include distributed training, real-time dashboards, and a single unified model that continuously improves with each dataset.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Training Modes](#-training-modes)
- [Training Dashboard](#-training-dashboard)
- [Distributed Training](#-distributed-training)
- [Model Architecture](#-model-architecture)
- [Configuration](#️-configuration)
- [Commands Reference](#-commands-reference)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Requirements](#-requirements)

---

## ✨ Features

### 🎯 Core Capabilities
- **Single Unified Model**: All training contributes to one model (`models/finai_gpt.pt`)
- **Continuous Learning**: Load and continue training from any checkpoint
- **Modern Architecture**: GPT-style transformer with RoPE, SwiGLU, Flash Attention
- **Optimized Training**: AdamW optimizer, cosine LR schedule, gradient accumulation
- **Accurate ETA**: Exponential moving average for smooth, reliable time estimates
- **Real-time Dashboard**: Beautiful web UI showing training metrics and progress

### 🌐 Distributed Training
- **Multi-Machine Training**: Train with friends across multiple computers
- **Automatic Synchronization**: Workers pull/push model checkpoints automatically
- **Web Dashboard**: Monitor all workers, tasks, and progress in real-time
- **No External Packages**: Distributed dashboard uses only Python stdlib

### 📊 Training Modes
1. **Single Dataset** (`train_single.py`): Quick training on one dataset
2. **Sequential** (`train_sequential.py`): Train datasets one-by-one with commits
3. **Batch** (`train_all.py`): Combine all pending datasets into one training run
4. **Distributed** (`distributed/`): Coordinate training across multiple machines

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd FinAI

# Install dependencies
pip install -r requirements.txt
```

### Train Your First Model

```bash
# Option 1: Train from a text file
python main.py train datasets/my_data.txt

# Option 2: Train from Hugging Face dataset
python main.py train_hf PatronusAI/financebench

# Option 3: Train on a single dataset with dashboard
python train_single.py <dataset-name>
```

### Chat with Your Model

```bash
python main.py chat
```

---

## 🎓 Training Modes

### 1. Single Dataset Training

Train on one Hugging Face dataset with automatic dashboard:

```bash
python train_single.py PatronusAI/financebench
```

**Features:**
- Automatic training dashboard at `http://localhost:8080`
- Real-time metrics: loss, ETA, progress
- Automatic CSV tracking (moves to `trained_datasets.csv`)
- Opens browser automatically

### 2. Sequential Training

Train datasets one-by-one from `datasets.csv`:

```bash
python train_sequential.py
```

**Features:**
- Processes each dataset individually
- Git commit after each dataset
- Skips already trained datasets
- Updates CSV status automatically

### 3. Batch Training

Combine all pending datasets and train once:

```bash
python train_all.py
```

**Features:**
- Merges all pending datasets into one file
- Single training run for efficiency
- Git commits for each dataset
- Automatic cleanup

### 4. Distributed Training

Train across multiple machines:

```bash
# On server (Raspberry Pi or always-on machine)
cd distributed
python server.py

# On each worker machine
python worker.py --server http://server-ip:8765

# Monitor with dashboard
python dashboard.py --server http://server-ip:8765
```

**Features:**
- Coordinate training across unlimited workers
- Automatic model synchronization
- Real-time monitoring dashboard
- Task queue management

📖 **[Full Distributed Training Guide](docs/QUICKSTART.md)**

---

## 📊 Training Dashboard

Every local training session automatically launches a beautiful web dashboard:

### Features
- **Live Metrics**: Loss, learning rate, step progress
- **Accurate ETA**: Exponential moving average for reliable estimates
- **Loss Chart**: Visual graph of last 50 steps
- **Configuration**: View all training parameters
- **Auto-refresh**: Updates every 10 seconds

### Access
- Automatically opens at `http://localhost:8080`
- Or run standalone: `python training_dashboard.py`

### Screenshot
```
┌─────────────────────────────────────────┐
│  🚀 FinAI Training Dashboard            │
│  Status: TRAINING                       │
├─────────────────────────────────────────┤
│  Progress: 2,500 / 5,000 (50.0%)       │
│  Elapsed: 0:15:30 | ETA: 0:15:30       │
│  Loss: 2.1045 | LR: 6.00e-04           │
│  Device: CUDA | Dataset: financebench   │
└─────────────────────────────────────────┘
```

---

## 🌐 Distributed Training

### Architecture

```
┌──────────────┐
│   Server     │  ← Coordinates tasks, stores model
│ (Raspberry Pi)│
└──────┬───────┘
       │
   ┌───┴────┬─────────┬─────────┐
   │        │         │         │
┌──▼──┐  ┌─▼───┐  ┌──▼──┐  ┌───▼──┐
│Worker│  │Worker│  │Worker│  │Worker│
│  #1  │  │  #2 │  │  #3 │  │  #4  │
└──────┘  └─────┘  └─────┘  └──────┘
```

### Setup

1. **Start Server** (on always-on machine):
```bash
cd distributed
python server.py
```

2. **Start Workers** (on each training machine):
```bash
python worker.py --server http://server-ip:8765
```

3. **Submit Tasks** (from any machine):
```bash
python client.py submit PatronusAI/financebench
```

4. **Monitor Progress**:
```bash
python dashboard.py --server http://server-ip:8765
```

### Key Features

- **Single Model**: All workers contribute to `models/finai_gpt.pt`
- **Auto-sync**: Workers download latest model before training
- **Fault Tolerant**: Failed tasks automatically reassigned
- **No Flask**: Dashboard uses only Python stdlib (http.server)

📖 **[Distributed Training Documentation](docs/README.md)**  
📖 **[Dashboard Guide](docs/DASHBOARD_GUIDE.md)**  
📖 **[Remote Access Setup](docs/REMOTE_ACCESS_SETUP.md)**

---

## 🏗️ Model Architecture

### Transformer Specifications

```python
Architecture: GPT-style Decoder-only Transformer
Parameters: ~15M (configurable)
Layers: 4
Attention Heads: 4
Embedding Dimension: 256
Context Window: 256 tokens
Vocabulary: ~50,000 tokens (BPE)
```

### Modern Features

- **RoPE (Rotary Position Embeddings)**: Better position encoding
- **SwiGLU Activation**: Improved over ReLU/GELU
- **Flash Attention**: 2-4x faster attention computation
- **Gradient Checkpointing**: 40% memory savings
- **Weight Tying**: Shared input/output embeddings

### Training Optimizations

- **AdamW Optimizer**: L2 regularization for better generalization
- **Cosine LR Schedule**: Smooth learning rate decay
- **Gradient Accumulation**: Simulate larger batch sizes
- **Mixed Precision (bf16)**: 50% memory reduction, full accuracy
- **Gradient Clipping**: Prevents training instability

---

## ⚙️ Configuration

All settings in `src/config.py`:

### Model Architecture
```python
N_LAYER = 4              # Transformer layers
N_HEAD = 4               # Attention heads  
N_EMBD = 256             # Embedding dimension
BLOCK_SIZE = 256         # Context window
DROPOUT = 0.05           # Dropout rate
```

### Training Parameters
```python
TRAIN_STEPS = 5000       # Training steps
BATCH_SIZE = 16          # Batch size
GRADIENT_ACCUM_STEPS = 4 # Gradient accumulation
LEARNING_RATE = 6e-4     # Learning rate
WEIGHT_DECAY = 0.1       # L2 regularization
WARMUP_STEPS = 100       # LR warmup steps
MAX_GRAD_NORM = 1.0      # Gradient clipping
```

### Generation Settings
```python
MAX_NEW_TOKENS = 512     # Max generation length
TEMPERATURE = 0.7        # Sampling temperature
TOP_K = 40               # Top-k sampling
TOP_P = 0.9              # Nucleus sampling
```

### Paths
```python
MODEL_DIR = "models"
LANGUAGE_MODEL_PATH = "models/finai_gpt.pt"  # Single unified model
TOKENIZER_PATH = "models/tokenizer.pkl"
DATASET_DIR = "datasets"
```

---

## 📝 Commands Reference

### Main CLI (`main.py`)

```bash
# Train from text file
python main.py train <file.txt> [--steps N] [--batch-size N] [--lr RATE]

# Train from Hugging Face dataset
python main.py train_hf <dataset-id> [--split train] [--max N]

# Interactive chat
python main.py chat

# Generate from prompt
python main.py generate "Your prompt here"
```

### Training Scripts

```bash
# Single dataset (with dashboard)
python train_single.py <hf-dataset-name>

# Sequential training
python train_sequential.py

# Batch training
python train_all.py
```

### Distributed Training

```bash
# Server
cd distributed
python server.py [--port 8765]

# Worker
python worker.py --server http://server:8765 [--name worker-1]

# Client (submit tasks)
python client.py submit <dataset-name>
python client.py status
python client.py workers

# Dashboard
python dashboard.py --server http://server:8765 [--port 8081]
```

### Dashboards

```bash
# Local training dashboard
python training_dashboard.py [--port 8080]

# Distributed dashboard
cd distributed
python dashboard.py [--server http://localhost:8765] [--port 8081]
```

---

## 📁 Project Structure

```
FinAI/
├── main.py                      # Main CLI entry point
├── train_single.py              # Single dataset training
├── train_sequential.py          # Sequential training
├── train_all.py                 # Batch training
├── run_prompt.py                # Quick generation script
├── training_dashboard.py        # Local training dashboard
├── requirements.txt             # Python dependencies
├── datasets.csv                 # Pending datasets
├── trained_datasets.csv         # Completed datasets
│
├── src/                         # Core source code
│   ├── core/
│   │   ├── finai.py            # Main FinAI class
│   │   └── context.py          # Conversation context
│   ├── models/
│   │   └── language_model_pytorch.py  # GPT model implementation
│   ├── data/
│   │   └── tokenizer.py        # BPE tokenizer
│   ├── config.py               # Configuration
│   └── training_metrics.py     # Metrics tracking
│
├── distributed/                 # Distributed training system
│   ├── server.py               # Coordination server
│   ├── worker.py               # Training worker
│   ├── client.py               # Task submission client
│   ├── dashboard.py            # Monitoring dashboard (no Flask!)
│   ├── server_config.json      # Server configuration
│   └── worker_config.json      # Worker configuration
│
├── scripts/                     # Utility scripts
│   ├── manage_datasets.py      # Dataset CSV management
│   └── export_hf_to_txt.py     # HF dataset export
│
├── models/                      # Model checkpoints
│   ├── finai_gpt.pt            # Unified model (single file)
│   └── tokenizer.pkl           # Tokenizer
│
├── datasets/                    # Training data
│   └── temp_*.txt              # Temporary training files
│
└── docs/                        # Documentation
    ├── README.md               # Distributed training docs
    ├── QUICKSTART.md           # Quick start guide
    ├── DASHBOARD_GUIDE.md      # Dashboard documentation
    ├── REMOTE_ACCESS_SETUP.md  # Remote access guide
    ├── EFFICIENCY_ANALYSIS.md  # Performance analysis
    └── IMPLEMENTATION_COMPLETE.md  # Implementation notes
```

---

## 📚 Documentation

### Core Documentation
- **[README](README.md)** - This file
- **[Configuration Guide](src/config.py)** - All configuration options

### Distributed Training
- **[Distributed Training Overview](docs/README.md)** - Complete distributed system guide
- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Dashboard Guide](docs/DASHBOARD_GUIDE.md)** - Monitoring and management
- **[Remote Access Setup](docs/REMOTE_ACCESS_SETUP.md)** - Configure remote access
- **[Efficiency Analysis](docs/EFFICIENCY_ANALYSIS.md)** - Performance benchmarks
- **[Implementation Notes](docs/IMPLEMENTATION_COMPLETE.md)** - Technical details

### Training Guides
- **[Training Loss Explained](docs/TRAINING_LOSS_EXPLAINED.md)** - Why loss goes up/down, what's normal

### Scripts Documentation
- **[Dataset Management](scripts/manage_datasets.py)** - CSV tracking system
- **[HF Export](scripts/export_hf_to_txt.py)** - Export Hugging Face datasets

---

## 📦 Requirements

### Core Dependencies
```
torch>=2.0.0              # PyTorch (CUDA/ROCm/CPU)
transformers>=4.30.0      # HF transformers (scheduler)
datasets>=2.14.0          # HF datasets
accelerate>=0.20.0        # Multi-GPU training (optional)
requests>=2.28.0          # HTTP requests (distributed)
```

### Optional Dependencies
```
torch-directml            # DirectML backend (AMD on Windows)
flash-attn               # Flash Attention (NVIDIA only)
```

### System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 2GB disk space

**Recommended:**
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (or AMD with ROCm)
- 10GB disk space

### Installation

```bash
# Basic installation
pip install -r requirements.txt

# With CUDA (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# With ROCm (AMD)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# With DirectML (AMD on Windows)
pip install torch-directml
```

---

## 🎯 Usage Examples

### Example 1: Quick Training

```bash
# Train on a financial dataset
python train_single.py PatronusAI/financebench

# Dashboard opens automatically at http://localhost:8080
# Watch real-time metrics: loss, ETA, progress
```

### Example 2: Batch Training

```bash
# Add datasets to datasets.csv
echo "PatronusAI/financebench,,train" >> datasets.csv
echo "FinGPT/fingpt-sentiment-train,,train" >> datasets.csv

# Train all at once
python train_all.py
```

### Example 3: Distributed Training

```bash
# On server (Raspberry Pi)
cd distributed
python server.py

# On worker machines (your PC + friends' PCs)
python worker.py --server http://raspberrypi.local:8765 --name my-pc

# Submit tasks from anywhere
python client.py submit PatronusAI/financebench
python client.py submit FinGPT/fingpt-sentiment-train

# Monitor at http://raspberrypi.local:8081
python dashboard.py --server http://raspberrypi.local:8765
```

### Example 4: Chat with Model

```bash
python main.py chat

# Or use the quick prompt script
python run_prompt.py
```

---

## 🔧 Troubleshooting

### Training Issues

**Problem**: Out of memory  
**Solution**: Reduce `BATCH_SIZE` or enable `USE_GRAD_CHECKPOINTING` in config

**Problem**: Slow training  
**Solution**: Enable GPU, use `--accelerate on`, increase `BATCH_SIZE`

**Problem**: NaN loss  
**Solution**: Reduce `LEARNING_RATE`, check `MAX_GRAD_NORM` is set

### Distributed Issues

**Problem**: Workers can't connect to server  
**Solution**: Check firewall, use correct IP/port, verify `AUTH_PASSWORD`

**Problem**: Model not syncing  
**Solution**: Ensure `models/finai_gpt.pt` exists on server, check permissions

**Problem**: Dashboard shows "offline"  
**Solution**: Verify server is running, check `SERVER_URL` in dashboard config

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **Hugging Face** - Transformers, Datasets, Accelerate
- **PyTorch** - Deep learning framework
- **OpenAI** - GPT architecture inspiration
- **Anthropic** - Modern training techniques

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/FinAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/FinAI/discussions)
- **Email**: your.email@example.com

---

**Built with ❤️ for the financial AI community**
