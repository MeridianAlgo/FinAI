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

## Distributed Training (Server/Worker)

FinAI supports distributed, CPU-only training across multiple machines using a lightweight server/worker architecture with dataset sharding and FedAvg aggregation. It includes robust error handling, persistence, and optional auth for remote workers.

### Components
- **Server**: `distributed_server.py`
  - Assigns shards to workers, tracks progress, aggregates checkpoints via weighted FedAvg.
  - Auto-sizes shard count based on active workers (heartbeats), with a configurable max.
  - Persists session state to disk so you can resume later.
  - Optional token auth for secure remote access.
- **Worker**: `distributed_worker.py`
  - Fetches shard jobs, trains CPU-only on its shard, uploads shard checkpoints.
  - Sends heartbeats using a `worker_id` for dynamic shard sizing.
  - Reports detailed errors with tracebacks and exits on failure.

### CSV Tracking
- `datasets.csv`: Pending datasets to train.
- `trained_datasets.csv`: Only successful trainings are recorded here.
- On failures, datasets remain in `datasets.csv` for retry; nothing is written to `trained_datasets.csv`.

### Start the Server
Install CPU-only PyTorch (aggregation uses torch):
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```
Run the server (token optional but recommended for remote access):
```bash
python distributed_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --max-shards 8 \
  --token YOUR_SECRET_TOKEN
```

Server endpoints:
- `GET /next_job?worker_id=<id>`: Assigns a shard to a worker.
- `POST /report_shard?name=<dataset>&shard_index=<i>&total_shards=<n>&num_samples=<k>` (binary body): Upload shard checkpoint.
- `POST /report_result`: Legacy/single-shard reporting.
- `GET /health`: Returns `{ ok: true, active_workers: N }`.

### Start Workers
Install deps:
```bash
pip install datasets numpy
pip install --index-url https://download.pytorch.org/whl/cpu torch
```
Run worker(s):
```bash
python distributed_worker.py \
  --server http://<server-ip>:8000 \
  --id <unique-worker-id> \
  --token YOUR_SECRET_TOKEN
```

Workers will:
- Poll `/next_job` (with `worker_id`) which doubles as a heartbeat.
- Train only their assigned shard on CPU and upload a checkpoint to the server.
- Exit with a clear error and traceback on failure.

### Dynamic Sharding and Leases
- The server automatically sets `total_shards` = min(active_workers, `--max-shards`).
- Sessions are persisted under `uploads/sessions/<dataset>.json` and include assigned/completed shards and leases.
- A shard assignment has a lease. If a worker disappears, the server reclaims the shard after a timeout and reassigns it.
- Once all shards complete, the server aggregates checkpoints with weighted FedAvg and saves:
  - `models/distributed/<dataset_slug>/finai_gpt_fedavg.pt`

Tip: For strict non-overlapping shards, connect the expected workers before a dataset starts (so shard count stabilizes). Dynamic increases mid-run may reuse some data; FedAvg still preserves accuracy.

### Error Handling (Server + Worker)
- All handlers wrapped in try/except with full traceback logging on the server.
- Worker prints and reports exact errors with traceback; exits on failure.
- Failed datasets are NOT added to `trained_datasets.csv` and are NOT removed from `datasets.csv`.

### Sequential Training With Fail-Fast
`train_sequential_v2.py` trains datasets one-by-one on CPU. If a dataset fails, it prints the exact error and exits immediately. Only successful datasets are moved to `trained_datasets.csv`.

Run:
```bash
python train_sequential_v2.py
```

### Remote Access Options
You can train from another network using either:

- Tailscale (recommended):
  1) Install on server and worker PCs. 2) `tailscale up` on each. 3) Use the server's tailnet IP in `--server`.

- Port Forwarding:
  - Forward TCP 8000 on your router to the server's LAN IP.
  - Use your public IP in `--server` and set a strong `--token`.

Examples:
```bash
# Tailscale example
python distributed_worker.py --server http://100.x.y.z:8000 --id home-worker-1 --token YOUR_SECRET_TOKEN

# Port forwarding example
python distributed_worker.py --server http://YOUR_PUBLIC_IP:8000 --id remote-worker-1 --token YOUR_SECRET_TOKEN
```

### Health and Monitoring
```bash
curl http://<server-ip>:8000/health
# -> { "ok": true, "active_workers": N }
```

### Artifacts
- Sequential: `models/<dataset_slug>/finai_gpt.pt`, `models/<dataset_slug>/tokenizer.pkl`
- Distributed aggregated: `models/distributed/<dataset_slug>/finai_gpt_fedavg.pt`
