# FinAI

Neural next-word language model focused on finance. Runs locally with no external LLM APIs.

## Features
- Train from local .txt datasets (one example per line)
- Train from Hugging Face datasets by ID, e.g. `npvinHnivqn/EnglishDictionary`
- **GPU acceleration support** for AMD and NVIDIA GPUs (via PyTorch)
- Optional streaming training for large datasets (memory efficient)
- Saves `models/tokenizer.pkl` and `models/language_model.pkl`
- Simple CLI for training, chatting, and generating text

## Project Structure
- `src/` — core code (model, tokenizer, data loader, app)
- `models/` — saved artifacts (`tokenizer.pkl`, `language_model.pkl`)
- `datasets/` — place local datasets here (created automatically)
- `docs/` — documentation (ARCHITECTURE.md, GPU_SETUP.md)
- `scripts/` — utility scripts (dataset preparation, GPU verification)
- `tests/` — tests and utilities (created automatically)
- `main.py` — CLI entrypoint
- `requirements.txt` — Python dependencies

## Install
1. Python 3.9+
2. Create venv (recommended)
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### GPU Acceleration Setup (Optional but Recommended)

For **AMD GPUs on Windows** (e.g., Radeon RX 7600XT):
```bash
# Option 1: PyTorch with ROCm (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Option 2: DirectML (simpler alternative)
pip install torch-directml
```

For **NVIDIA GPUs**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU setup:**
```bash
python scripts/verify_gpu.py
```

See `docs/GPU_SETUP.md` for detailed instructions.

## Usage

### Quick Start - Sequential Training (Recommended)

Train on each dataset individually, one at a time:

1. **Edit `datasets_list.py`** to add your datasets:
   ```python
   DATASETS = [
       ("virattt/financial-qa-10K", None, None),
       ("AdaptLLM/finance-tasks", "ConvFinQA", None),
       # Add more datasets here...
   ]
   ```

2. **Run the training script**:
   ```bash
   python train_sequential.py
   ```

This will:
- Train on each dataset individually
- Remove each dataset from the list after training completes
- Auto-detect GPU (or use CPU)
- Save progress - you can stop and resume anytime

### Alternative - Train All Datasets Combined

Train on all datasets combined into one training file:

```bash
python train_all_datasets.py
```

This script will:
1. Download all financial datasets automatically
2. Combine them into one training file
3. Auto-detect your GPU (or use CPU)
4. Train the model on the combined data

### Quick Start - Interactive Mode

The easiest way to train is using interactive mode (no command-line flags needed!):

```bash
python main.py
```

Or directly:
```bash
python main.py interactive
```

This will prompt you for:
- **Dataset name**: e.g., `AdaptLLM/finance-tasks`
- **Dataset type/split**: e.g., `FPB` (or press Enter for 'train')
- **Sample size**: Optional (or press Enter for default)
- **Streaming mode**: y/n (default: y)

The system will **automatically detect your GPU** (ROCm/DirectML/CUDA) and use it if available, otherwise use CPU.

### 1) Train from a local .txt file
Each line is a training example.
```bash
# GPU auto-detected (uses GPU if available, CPU otherwise)
python main.py train datasets/your_data.txt

# Force CPU-only training
python main.py train datasets/your_data.txt --no-gpu
```

### 2) Train from a Hugging Face dataset ID
Pass only the dataset ID (no import needed). Example:
```bash
python main.py train_hf npvinHnivqn/EnglishDictionary
```
Options:
- `--split <name>`: dataset split (default: `train`)
- `--sample <N>`: limit examples (helps quick runs)
- `--field <col>`: specify text column (auto-detected if omitted)
- `--stream`: stream and train incrementally (best for large datasets)
- `--use-gpu`: force GPU acceleration (GPU auto-detected by default)
- `--no-gpu`: force CPU-only training

Examples:
```bash
# Quick smoke test (small sample, streaming)
python main.py train_hf npvinHnivqn/EnglishDictionary --sample 1000 --stream

# Full in-memory training (small datasets)
python main.py train_hf npvinHnivqn/EnglishDictionary --sample 5000
```

### 3) Generate text from a prompt
Requires trained artifacts in `models/`.
```bash
python main.py generate "the stock market"
```

### 4) Interactive chat
```bash
python main.py chat
```

## Tips for Speed and Quality
- **GPU Acceleration**: Enable GPU acceleration for 5-20x faster training. See `docs/GPU_SETUP.md` for setup instructions.
- Use `--stream` for datasets larger than a few MB. It avoids loading everything in memory and uses incremental SGD training.
- Adjust `EPOCHS`, `BATCH_SIZE`, and `VOCAB_SIZE` in `src/config.py`. GPU training benefits from larger batch sizes (2048-4096).
- Keep `MAX_SEQUENCE_LENGTH` modest (e.g., 50) to reduce compute and memory usage.
- For finance-specific quality, curate domain-focused corpora (e.g., filings, news, reports). Clean text and remove boilerplate.
- Consider multiple runs with different seeds and ensembling if needed.

## Troubleshooting
- **GPU not detected**: Run `python scripts/verify_gpu.py` to check GPU setup. See `docs/GPU_SETUP.md` for troubleshooting.
- If `datasets` library cannot download, ensure internet access and run `python -m pip install datasets`.
- If generation says model not loaded, ensure you trained first and `models/` contains the pickled files.
- **Out of memory errors**: Reduce `BATCH_SIZE` in `src/config.py` or use `--stream` for streaming training.

## License
MIT (adjust as desired)
