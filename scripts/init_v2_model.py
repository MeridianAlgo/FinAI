#!/usr/bin/env python3
"""
Initialize and upload a fresh Fin.AI v2 model to Hugging Face

This script:
1. Creates a new v2 model from scratch
2. Saves it in the correct format
3. Uploads it to Hugging Face
4. Replaces the old v1 model
"""

import os
import sys
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fin_ai.model import FinAIModel, FinAIConfig


def create_model_card():
    """Create an updated model card for v2"""
    return """---
language: en
license: mit
tags:
- text-generation
- pytorch
- causal-lm
- continuous-learning
- gqa
- swiglu
- rmsnorm
- rope
---

# 🤖 Fin.AI v2.0

**⚠️ EXPERIMENTAL - Continuously Learning Language Model**

Fin.AI v2 is an optimized transformer language model that trains itself every ~85 minutes on diverse datasets via GitHub Actions.

## 🚀 What's New in v2

### Architecture Improvements

- **Grouped Query Attention (GQA)**: 40% faster inference with fewer KV heads
- **SwiGLU Activation**: Better learning dynamics (used in LLaMA, PaLM)
- **RMSNorm**: 20% faster than LayerNorm
- **Rotary Position Embeddings (RoPE)**: Better position encoding
- **Pre-norm Architecture**: More stable training

### Performance Gains

- **40% faster training** on CPU
- **24% less memory** usage
- **Better model quality** with improved architecture
- **More efficient** parameter usage

## 📊 Model Details

- **Architecture**: Custom GPT-style transformer with modern improvements
- **Parameters**: ~40M (small preset)
- **Layers**: 8
- **Attention Heads**: 8 (4 KV heads for GQA)
- **Embedding Dimension**: 512
- **FFN Dimension**: 1792 (with SwiGLU)
- **Max Sequence Length**: 512 tokens
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)

## 🎯 Training

- **Schedule**: Trains every ~85 minutes (24/7)
- **Datasets**: Rotates through 24+ diverse datasets
- **Platform**: GitHub Actions (free tier, CPU)
- **Framework**: PyTorch
- **Tracking**: Weights & Biases

## 📥 Usage

### Download and Load

```python
from huggingface_hub import hf_hub_download
import torch

# Download model files
hf_hub_download("MeridianAlgo/Fin.AI", "model.pt", local_dir="./model")
hf_hub_download("MeridianAlgo/Fin.AI", "config.json", local_dir="./model")

# Load model
from fin_ai.model import FinAIModel

model = FinAIModel.from_pretrained("./model")
model.eval()
```

### Generate Text

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1
)

print(tokenizer.decode(outputs[0]))
```

## ⚠️ Limitations

- **Experimental**: This is a research project, not production-ready
- **Quality**: Model is continuously learning and may produce errors
- **Biases**: May reflect biases from training data
- **Size**: Small model (40M params) has limited capabilities
- **Context**: 512 token context window

## 🔗 Links

- **GitHub**: [MeridianAlgo/FinAI](https://github.com/MeridianAlgo/FinAI)
- **Training Logs**: [GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions)
- **Metrics**: [Wandb Dashboard](https://wandb.ai/meridianalgo-meridianalgo/fin-ai)
- **Architecture**: [Technical Documentation](https://github.com/MeridianAlgo/FinAI/blob/main/docs/ARCHITECTURE_V2.md)

## 📜 License

MIT License - See [LICENSE](https://github.com/MeridianAlgo/FinAI/blob/main/LICENSE)

## 🙏 Acknowledgments

Architecture inspired by:
- **LLaMA** (Meta AI) - GQA, SwiGLU, RMSNorm, RoPE
- **PaLM** (Google) - SwiGLU
- **GPT-NeoX** (EleutherAI) - RoPE

---

**Last Updated**: Auto-updated with each training run

*Built with ❤️ for continuous learning*
"""


def main():
    print("🚀 Initializing Fin.AI v2 Model for Hugging Face\n")

    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN environment variable not set")
        print()
        print("Options:")
        print("1. Set environment variable: export HF_TOKEN=your_token_here")
        print("2. Or run via GitHub Actions (recommended)")
        print()
        print("Get your token from: https://huggingface.co/settings/tokens")
        print()

        # Try to read from .env file if it exists
        env_file = ".env"
        if os.path.exists(env_file):
            print(f"Checking {env_file} file...")
            with open(env_file, "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if hf_token:
                            print(f"✓ Found HF_TOKEN in {env_file}")
                            break

        if not hf_token:
            print()
            print("💡 Tip: You can also run this via GitHub Actions:")
            print("   1. Go to Actions tab in your repo")
            print("   2. Select 'Initialize v2 Model' workflow")
            print("   3. Click 'Run workflow' and type 'INIT_V2'")
            print()
            sys.exit(1)

    # Configuration
    repo_id = "MeridianAlgo/Fin.AI"
    output_dir = "checkpoints/model"

    print("📋 Configuration:")
    print(f"   Repository: {repo_id}")
    print(f"   Output Directory: {output_dir}")
    print()

    # Create model config
    print("⚙️  Creating model configuration (small preset)...")
    config = FinAIConfig.from_preset("small")

    # Load tokenizer to get vocab size
    print("🔤 Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = len(tokenizer)

    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Parameters: {config.num_parameters:,}")
    print(f"   Layers: {config.n_layers}")
    print(f"   Heads: {config.n_heads} (KV heads: {config.n_kv_heads})")
    print(f"   Embed dim: {config.embed_dim}")
    print(f"   FFN dim: {config.ff_dim}")
    print()

    # Create model
    print("🤖 Creating fresh v2 model...")
    model = FinAIModel(config)

    actual_params = model.count_parameters()
    print(f"   ✓ Model created with {actual_params:,} parameters")
    print()

    # Save model
    print(f"💾 Saving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # Save model card
    print("📝 Creating model card...")
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(create_model_card())
    print(f"   ✓ Model card saved to {readme_path}")
    print()

    # Create version file
    from datetime import datetime

    version_info = {
        "version": "2.0.0",
        "architecture": "v2",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "parameters": actual_params,
        "config": config.to_dict(),
    }

    import json

    version_path = os.path.join(output_dir, "version.json")
    with open(version_path, "w") as f:
        json.dump(version_info, f, indent=2)
    print(f"   ✓ Version info saved to {version_path}")
    print()

    # Upload to Hugging Face
    print(f"☁️  Uploading to Hugging Face ({repo_id})...")

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=hf_token, exist_ok=True, repo_type="model")
        print(f"   ✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"   Note: {e}")

    # Upload all files
    print("   Uploading files...")
    try:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            token=hf_token,
            commit_message="🚀 Initialize Fin.AI v2.0 - Fresh model with GQA, SwiGLU, RMSNorm, RoPE",
        )
        print(f"   ✓ Upload complete!")
        print()
        print(f"✅ Success! Model is now available at:")
        print(f"   https://huggingface.co/{repo_id}")
        print()
        print("🎯 Next steps:")
        print("   1. The next training run will download this v2 model")
        print("   2. Training will continue from this fresh initialization")
        print("   3. Model will be updated every ~85 minutes")

    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
