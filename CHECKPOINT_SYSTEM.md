# Fin.AI Checkpoint System

## Overview

Fin.AI uses a sophisticated checkpoint system that syncs training state between local/CI environments and Hugging Face Hub. This enables continuous training across multiple runs while maintaining dataset-specific progress.

## How It Works

### 1. Checkpoint Structure

Each checkpoint contains:
- **Model weights** (`model_state_dict`)
- **Optimizer state** (`optimizer_state_dict`)
- **Scheduler state** (`scheduler_state_dict`)
- **Training progress** (`global_step`, `epoch`)
- **Dataset identifier** (`dataset`)

Checkpoints are named: `checkpoint-{dataset_name}-{step}.pt`

Example: `checkpoint-fineweb-edu-1000.pt`

### 2. Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Pull Latest Checkpoint from Hugging Face                │
│    - Looks for checkpoint-{dataset}-*.pt files              │
│    - Downloads the latest checkpoint for current dataset    │
│    - Also downloads model weights if available              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Load Checkpoint (Priority Order)                         │
│    a) Dataset-specific checkpoint (checkpoint-{dataset}-*.pt)│
│    b) Model weights only (model.safetensors)                │
│    c) Fresh initialization                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Train for N Steps                                        │
│    - Saves checkpoint every save_steps (default: 800)       │
│    - Each save creates a new checkpoint file                │
│    - Old checkpoints are cleaned up (keeps last 2)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Push Checkpoint to Hugging Face                          │
│    - Uploads checkpoint-{dataset}-{step}.pt                 │
│    - Uploads model weights (model.safetensors)              │
│    - Includes commit message with timestamp                 │
└─────────────────────────────────────────────────────────────┘
```

### 3. Dataset-Specific Checkpoints

The system tracks progress separately for each dataset:

- **fineweb-edu**: `checkpoint-fineweb-edu-1000.pt`
- **TinyStories**: `checkpoint-TinyStories-500.pt`
- **OpenWebText**: `checkpoint-OpenWebText-2000.pt`

This allows the model to:
- Resume training on the same dataset from where it left off
- Start fresh when switching to a new dataset (using existing weights)
- Maintain separate training states for different data sources

### 4. Configuration

In `config/model_config.yaml`:

```yaml
checkpointing:
  output_dir: "./checkpoints"
  save_total_limit: 2  # Keep only last 2 checkpoints locally
  resume_from_checkpoint: true
  hf_repo_id: "MeridianAlgo/Fin.AI"
  push_to_hub: true  # Enable HF sync
```

### 5. Authentication

The system looks for `HF_TOKEN` in two places:

1. **Environment variable**: `os.environ.get("HF_TOKEN")` (used in CI/CD)
2. **.env file**: Reads from `.env` in project root (local development only)

**Local Development** - Create `.env` file:
```bash
# .env (DO NOT COMMIT THIS FILE!)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
COMET_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

**CI/CD (GitHub Actions)** - Use repository secrets:
- Go to Settings → Secrets and variables → Actions
- Add secrets: `HF_TOKEN`, `COMET_API_KEY`, etc.
- The workflow passes these as environment variables

**Important**: 
- ✅ `.env` is in `.gitignore` - never commit it!
- ✅ Use GitHub Secrets for CI/CD
- ✅ The code checks environment variables first, then falls back to `.env`

## GitHub Actions Integration

The workflow (`.github/workflows/train.yml`) handles checkpoints automatically:

1. **Before Training**:
   - Passes secrets as environment variables (HF_TOKEN, COMET_API_KEY)
   - Downloads latest model from HF
   - Trainer pulls latest checkpoint from HF

2. **During Training**:
   - Saves checkpoints every 800 steps
   - Pushes each checkpoint to HF immediately

3. **After Training**:
   - Uploads final model weights to HF
   - Commits checkpoint files to repo
   - Creates release every 15 runs

**Security**: All secrets are stored in GitHub repository secrets, never in code or `.env` files in the repo.

## Local Development

### First Time Setup

```bash
# 1. Create .env file
echo "HF_TOKEN=your_token_here" > .env

# 2. Run training
python train.py --config config/model_config.yaml --datasets config/datasets.yaml
```

### Resuming Training

The trainer automatically:
1. Pulls latest checkpoint from HF (if `push_to_hub: true`)
2. Loads the checkpoint for current dataset
3. Continues from last step

### Disabling HF Sync

To train locally without HF sync:

```yaml
checkpointing:
  push_to_hub: false  # Disable HF sync
```

## Checkpoint Files

### Local Files (in `checkpoints/`)

- `checkpoint-{dataset}-{step}.pt` - Full training state
- `model/model.safetensors` - Model weights only
- `model/config.json` - Model configuration
- `dataset_state.json` - Dataset cycling state
- `version.json` - Training run metadata

### Hugging Face Files

- `checkpoint-{dataset}-{step}.pt` - Full training state
- `model.safetensors` - Model weights
- `config.json` - Model configuration
- `README.md` - Model card
- `generation_config.json` - Generation settings

## Troubleshooting

### "HF_TOKEN not found" Warning

**Cause**: Token not in environment or `.env` file

**Fix**:
```bash
# Create .env file
echo "HF_TOKEN=hf_xxxxx" > .env

# Or set environment variable
export HF_TOKEN=hf_xxxxx
```

### "No checkpoint found on Hugging Face"

**Cause**: First training run or new dataset

**Expected**: The trainer will:
1. Load model weights if available
2. Start training from step 0
3. Create initial checkpoint
4. Push to HF

### Checkpoint Not Syncing

**Check**:
1. `push_to_hub: true` in config
2. `hf_repo_id` is set correctly
3. HF_TOKEN has write access to repo
4. Internet connection is stable

## Best Practices

1. **Always enable HF sync in CI/CD**: Ensures checkpoints persist across runs
2. **Keep save_total_limit low**: Saves disk space (2-3 is recommended)
3. **Use dataset-specific checkpoints**: Allows proper resume for each dataset
4. **Monitor HF repo size**: Clean up old checkpoints periodically
5. **Test locally first**: Verify checkpoint system works before CI/CD

## Advanced: Manual Checkpoint Management

### Download Specific Checkpoint

```python
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="MeridianAlgo/Fin.AI",
    filename="checkpoint-fineweb-edu-1000.pt",
    token="hf_xxxxx"
)
```

### Upload Checkpoint Manually

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="checkpoints/checkpoint-fineweb-edu-1000.pt",
    path_in_repo="checkpoint-fineweb-edu-1000.pt",
    repo_id="MeridianAlgo/Fin.AI",
    token="hf_xxxxx"
)
```

### Clean Up Old Checkpoints

```python
from huggingface_hub import HfApi

api = HfApi(token="hf_xxxxx")
files = api.list_repo_files("MeridianAlgo/Fin.AI")

# Delete old checkpoints
for f in files:
    if f.startswith("checkpoint-") and f.endswith(".pt"):
        # Keep only latest 5
        # ... your logic here
        api.delete_file(f, repo_id="MeridianAlgo/Fin.AI")
```

## Summary

The checkpoint system provides:
- ✅ Automatic sync between local and HF
- ✅ Dataset-specific progress tracking
- ✅ Continuous training across runs
- ✅ Easy resume from any point
- ✅ Minimal manual intervention

Just set your `HF_TOKEN` and let the system handle the rest!
