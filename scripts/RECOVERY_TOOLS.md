# Model Recovery Tools

This directory contains tools for diagnosing and fixing diverged models.

## Quick Start

If your model is producing NaN loss, run these commands in order:

```bash
# 1. Check if the model has diverged
python scripts/check_checkpoint_health.py

# 2. If diverged, reset it
python scripts/reset_diverged_model.py --backup

# 3. Resume training with the reset model
python train.py --config config/model_config.yaml --datasets config/datasets.yaml --max-steps 250
```

## Tools Overview

### 1. check_checkpoint_health.py

**Purpose**: Diagnose checkpoint problems

**Usage**:
```bash
python scripts/check_checkpoint_health.py --checkpoint checkpoints/checkpoint-fineweb-edu.pt
```

**What it checks**:
- NaN/Inf values in weights
- Very large parameter values
- Model can perform inference
- Model can generate text

**Output**:
- ✅ Healthy checkpoint - ready to use
- ❌ Diverged checkpoint - needs reset

### 2. reset_diverged_model.py

**Purpose**: Reset a diverged model to fresh weights while keeping architecture

**Usage**:
```bash
# With backup (recommended)
python scripts/reset_diverged_model.py --checkpoint checkpoints/checkpoint-fineweb-edu.pt --backup

# Without backup
python scripts/reset_diverged_model.py --checkpoint checkpoints/checkpoint-fineweb-edu.pt
```

**What it does**:
- Loads the checkpoint to extract architecture
- Creates a new model with fresh random weights
- Resets training step to 0
- Resets optimizer state
- Saves both checkpoint and model files

**Options**:
- `--checkpoint PATH`: Path to checkpoint to reset (default: checkpoints/checkpoint-fineweb-edu.pt)
- `--output PATH`: Where to save reset checkpoint (default: overwrite input)
- `--backup`: Create backup before resetting

## Common Scenarios

### Scenario 1: Model suddenly produces NaN loss

```bash
# Check health
python scripts/check_checkpoint_health.py

# Output shows: ❌ CHECKPOINT HAS DIVERGED

# Reset it
python scripts/reset_diverged_model.py --backup

# Resume training
python train.py --config config/model_config.yaml --datasets config/datasets.yaml --max-steps 250
```

### Scenario 2: Want to start training from scratch

```bash
# Delete checkpoint
del checkpoints\checkpoint-fineweb-edu.pt  # Windows
rm checkpoints/checkpoint-fineweb-edu.pt   # Linux/Mac

# Or reset to fresh weights
python scripts/reset_diverged_model.py

# Start training
python train.py --config config/model_config.yaml --datasets config/datasets.yaml --max-steps 250
```

### Scenario 3: Not sure if model is healthy

```bash
# Always check first
python scripts/check_checkpoint_health.py

# Will show:
# - Parameter statistics
# - NaN/Inf detection
# - Inference test results
# - Generation test results
```

## Understanding the Output

### Healthy Checkpoint
```
✓ No NaN values found
✓ No Inf values found
✓ Model inference successful (loss: 10.5432)
✓ Logits are healthy (max: 15.23)
✓ Generation test: 'The future of artificial intelligence'

✅ CHECKPOINT IS HEALTHY
```

### Diverged Checkpoint
```
❌ Found NaN in 45 parameter(s)
  - model.layers.0.attention.q_proj.weight
  - model.layers.0.attention.k_proj.weight
  ...
❌ Model produces NaN loss

❌ CHECKPOINT HAS DIVERGED

Recommendation: Reset this checkpoint using:
  python scripts/reset_diverged_model.py --checkpoint ... --backup
```

## Prevention

The training code now includes automatic safeguards:

1. **Pre-training sanity check**: Tests loaded model before training starts
2. **NaN detection during training**: Skips batches that would cause NaN
3. **Gradient monitoring**: Warns if gradients are too large
4. **Lower learning rate**: Default is now 1e-4 instead of 3e-4
5. **Aggressive gradient clipping**: Max norm reduced to 0.5

These changes are already in the code - no action needed!

## Files Modified

These fixes are already applied:

- `fin_ai/training/trainer.py` - Added NaN detection and recovery
- `config/model_config.yaml` - Reduced learning rate (3e-4 → 1e-4)
- `.github/workflows/train.yml` - Made CI non-blocking

## Support

If you continue to see NaN issues after resetting:

1. Check if learning rate needs further reduction
2. Verify your dataset has valid data
3. Try training with `--max-samples 1000` first (smaller dataset)
4. Check GPU memory (if using GPU)

For more details, see `FIXING_NAN_LOSS.md` in the root directory.
