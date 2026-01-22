# Fin.AI Training Flow

## Complete Training Cycle

This document describes the complete training flow from checkpoint download to upload.

## GitHub Actions Training Flow

### 1. Pre-Training: Download from Hugging Face

**Step**: `Download model and checkpoint from Hugging Face`

Downloads both model weights and training checkpoints:

```
📦 Hugging Face Repository
├── Model Files (checkpoints/model/)
│   ├── config.json ✅ (required)
│   ├── model.safetensors ✅
│   ├── configuration_finai.py
│   ├── modeling_finai.py
│   ├── generation_config.json
│   └── README.md
│
└── Checkpoint Files (checkpoints/)
    ├── checkpoint-fineweb-edu-0.pt
    ├── checkpoint-fineweb-edu-1000.pt
    └── checkpoint-fineweb-edu-2000.pt (latest)
```

**What happens:**
1. Lists all files in HF repository
2. Downloads required model files to `checkpoints/model/`
3. Finds latest checkpoint file (highest step number)
4. Downloads latest checkpoint to `checkpoints/`
5. If no checkpoint found, training starts from model weights only

### 2. Training: Continuous Learning

**Step**: `Train model`

The trainer (`fin_ai/training/trainer.py`) handles training:

```python
# Training loop
while global_step < max_steps:
    # Train for N steps
    train_batch()
    
    # Save checkpoint every save_steps (800)
    if global_step % save_steps == 0:
        save_checkpoint()
        push_checkpoint_to_hf()  # ← Pushes during training!
```

**Checkpoint behavior:**

1. **Load checkpoint** (priority order):
   - Dataset-specific checkpoint from HF (e.g., `checkpoint-fineweb-edu-2000.pt`)
   - Model weights only (starts from step 0)
   - Fresh initialization

2. **Train for 1000 steps** (configurable)

3. **Save checkpoints every 800 steps**:
   - Creates `checkpoint-{dataset}-{step}.pt` locally
   - Immediately pushes to Hugging Face
   - Keeps only last 2 checkpoints locally (saves disk space)

4. **Final checkpoint** at end of training

### 3. Post-Training: Upload to Hugging Face

**Step**: `Upload model to Hugging Face`

Uploads both model weights and all checkpoint files:

```
📤 Upload to Hugging Face
├── Model Weights (from checkpoints/model/)
│   ├── model.safetensors
│   ├── config.json
│   ├── configuration_finai.py
│   ├── modeling_finai.py
│   └── README.md
│
└── Checkpoint Files (from checkpoints/)
    ├── checkpoint-fineweb-edu-2000.pt (old)
    └── checkpoint-fineweb-edu-3000.pt (new) ← Latest after training
```

**What happens:**
1. Prepares model directory with all required files
2. Updates config.json with auto_map for trust_remote_code
3. Uploads model weights folder to HF
4. Uploads all checkpoint files (*.pt) to HF
5. Cleans up legacy V2 files if they exist

### 4. Version Control: Commit to GitHub

**Step**: `Update version and push`

Commits checkpoint files to GitHub repository:

```bash
git add checkpoints/checkpoint-*.pt
git add checkpoints/version.json
git add checkpoints/dataset_state.json
git commit -m "Training run #123 - FineWeb-Edu - 2026-01-22"
git push origin main
```

**Note**: Only checkpoint metadata is committed, not model weights (too large).

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DOWNLOAD FROM HUGGING FACE                               │
│    - Model weights (checkpoints/model/)                     │
│    - Latest checkpoint (checkpoint-{dataset}-{step}.pt)     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LOAD CHECKPOINT                                           │
│    - Trainer loads checkpoint-{dataset}-{step}.pt           │
│    - Resumes from last step (e.g., step 2000)               │
│    - Loads optimizer and scheduler state                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TRAIN FOR 1000 STEPS                                      │
│    - Step 2000 → 3000                                        │
│    - Saves checkpoint at step 2800 → pushes to HF           │
│    - Saves checkpoint at step 3000 → pushes to HF           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. UPLOAD TO HUGGING FACE                                    │
│    - Model weights (model.safetensors, config.json, etc.)   │
│    - All checkpoint files (checkpoint-*.pt)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. COMMIT TO GITHUB                                          │
│    - Checkpoint files (checkpoint-*.pt)                     │
│    - Version metadata (version.json)                        │
│    - Dataset state (dataset_state.json)                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Continuous Training
- Each run resumes from the last checkpoint
- No training progress is lost
- Model improves continuously across runs

### Dual Sync
- **Hugging Face**: Full model + checkpoints (for inference and training)
- **GitHub**: Checkpoint metadata (for version control)

### Dataset-Specific Checkpoints
- Each dataset has its own checkpoint file
- Switching datasets starts fresh but uses existing weights
- Example:
  - `checkpoint-fineweb-edu-3000.pt` (3000 steps on FineWeb-Edu)
  - `checkpoint-TinyStories-500.pt` (500 steps on TinyStories)

### Automatic Cleanup
- Keeps only last 2 checkpoints locally (saves disk space)
- All checkpoints preserved on Hugging Face
- Legacy V2 files automatically removed

## Configuration

### Enable/Disable HF Sync

In `config/model_config.yaml`:

```yaml
checkpointing:
  hf_repo_id: "MeridianAlgo/Fin.AI"
  push_to_hub: true  # Set to false to disable HF sync
  save_steps: 800    # Save checkpoint every N steps
  save_total_limit: 2  # Keep only last N checkpoints locally
```

### GitHub Secrets Required

- `HF_TOKEN`: Hugging Face API token (write access)
- `COMET_API_KEY`: Comet ML API key (optional)

## Troubleshooting

### Checkpoint Not Loading

**Symptom**: Training starts from step 0 every time

**Causes**:
1. No checkpoint file on HF
2. Checkpoint file corrupted
3. Dataset name mismatch

**Solution**:
```bash
# Check HF repository for checkpoint files
# Should see: checkpoint-{dataset}-{step}.pt
```

### Upload Failing

**Symptom**: "Failed to upload checkpoint" error

**Causes**:
1. HF_TOKEN not set or invalid
2. No write access to repository
3. Network issues

**Solution**:
1. Verify HF_TOKEN in GitHub Secrets
2. Check repository permissions
3. Retry workflow

### Disk Space Issues

**Symptom**: "No space left on device"

**Causes**:
1. Too many checkpoints locally
2. Large model size

**Solution**:
1. Reduce `save_total_limit` in config
2. Increase `save_steps` (save less frequently)
3. Use smaller model preset

## Best Practices

1. **Always enable HF sync in CI/CD**: Ensures checkpoints persist across runs
2. **Keep save_total_limit low**: Saves disk space (2-3 recommended)
3. **Monitor HF repository size**: Clean up old checkpoints periodically
4. **Use dataset-specific checkpoints**: Allows proper resume for each dataset
5. **Test locally first**: Verify checkpoint system works before CI/CD

## Summary

The training flow ensures:
- ✅ Checkpoints are downloaded before training
- ✅ Training resumes from last checkpoint
- ✅ Checkpoints are pushed during training (every 800 steps)
- ✅ Final model and checkpoints uploaded after training
- ✅ Version control via GitHub commits
- ✅ No training progress is lost
- ✅ Continuous improvement across runs

Every training run builds on the previous one, creating a continuously improving model!
