# Fin.AI v2 Initialization Guide

This guide explains how to initialize the new v2 model architecture on Hugging Face, replacing the old v1 model.

## Why Initialize?

The v2 architecture is incompatible with v1 checkpoints. To ensure smooth continuous training with the new architecture, we need to:

1. Create a fresh v2 model
2. Upload it to Hugging Face
3. Let future training runs continue from this v2 checkpoint

## Option 1: Initialize via GitHub Actions (Recommended)

This is the easiest method and doesn't require local setup.

### Steps:

1. Go to your GitHub repository
2. Click on **Actions** tab
3. Select **Initialize v2 Model** workflow
4. Click **Run workflow**
5. Type `INIT_V2` in the confirmation field
6. Click **Run workflow** button

The workflow will:
- Create a fresh v2 model (~40M parameters)
- Upload it to Hugging Face
- Replace the old v1 model

**Time**: ~2-3 minutes

## Option 2: Initialize Locally

If you prefer to run it locally or want to verify the model first.

### Prerequisites:

1. Python 3.10+ installed
2. Dependencies installed: `pip install -r requirements.txt`
3. Hugging Face token with write access

### Steps:

#### On Linux/Mac:

```bash
# Set your Hugging Face token
export HF_TOKEN=your_token_here

# Run initialization script
chmod +x init_v2_local.sh
./init_v2_local.sh
```

#### On Windows:

```cmd
# Set your Hugging Face token
set HF_TOKEN=your_token_here

# Run initialization script
init_v2_local.bat
```

#### Or directly with Python:

```bash
export HF_TOKEN=your_token_here  # Linux/Mac
# or
set HF_TOKEN=your_token_here     # Windows

python scripts/init_v2_model.py
```

**Time**: ~2-3 minutes

## What Gets Created?

The initialization creates:

```
checkpoints/model/
├── model.pt          # v2 model weights (~160MB)
├── config.json       # v2 model configuration
├── version.json      # Version metadata
└── README.md         # Model card for Hugging Face
```

## Model Specifications

The initialized v2 model has:

- **Architecture**: Custom transformer with GQA, SwiGLU, RMSNorm, RoPE
- **Parameters**: ~40M (small preset)
- **Layers**: 8
- **Attention Heads**: 8 (4 KV heads)
- **Embedding Dimension**: 512
- **FFN Dimension**: 1792
- **Max Sequence Length**: 512 tokens
- **Vocabulary**: 50,257 (GPT-2 tokenizer)

## Verification

After initialization, verify the model on Hugging Face:

1. Visit: https://huggingface.co/MeridianAlgo/Fin.AI
2. Check that files exist:
   - `model.pt`
   - `config.json`
   - `version.json`
   - `README.md`
3. Check the README shows "v2.0" and mentions GQA, SwiGLU, etc.

## Next Training Run

After initialization:

1. The next scheduled training run will automatically download the v2 model
2. Training will continue from this fresh v2 checkpoint
3. Model will be updated every ~85 minutes as usual

## Troubleshooting

### "HF_TOKEN not set"

Make sure you've set your Hugging Face token:

```bash
export HF_TOKEN=hf_...  # Get from https://huggingface.co/settings/tokens
```

### "Upload failed"

Check that:
- Your token has write access
- You have permission to push to MeridianAlgo/Fin.AI
- Your internet connection is stable

### "Model files not found"

The script creates files in `checkpoints/model/`. If they're missing:
- Check for error messages during model creation
- Ensure you have write permissions in the directory
- Try running with `python -u scripts/init_v2_model.py` for unbuffered output

## Manual Verification

To test the model locally before uploading:

```python
from fin_ai.model import FinAIModel

# Load the created model
model = FinAIModel.from_pretrained("checkpoints/model")

# Check it works
import torch
input_ids = torch.randint(0, 50257, (1, 10))
outputs = model(input_ids)
print(f"✓ Model works! Output shape: {outputs['logits'].shape}")
```

## Rollback (if needed)

If you need to rollback to v1:

1. In `fin_ai/model/__init__.py`, change:
   ```python
   from fin_ai.model.transformer import FinAIModel as FinAIModelLegacy
   # Use legacy as default
   FinAIModel = FinAIModelLegacy
   ```

2. Re-run training

## Questions?

- Check [GitHub Issues](https://github.com/MeridianAlgo/FinAI/issues)
- See [Migration Guide](legacy/MIGRATION_V2.md)
- Review [Architecture Docs](docs/ARCHITECTURE_V2.md)

---

**Ready to initialize?** Choose Option 1 (GitHub Actions) or Option 2 (Local) above! 🚀
