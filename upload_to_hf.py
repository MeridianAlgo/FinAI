#!/usr/bin/env python3
"""
Upload existing model to Hugging Face Hub

Usage:
    python upload_to_hf.py --token YOUR_HF_TOKEN
    
Or set HF_TOKEN environment variable:
    export HF_TOKEN=your_token
    python upload_to_hf.py
"""

import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Upload Fin.AI model to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--repo", type=str, default="MeridianAlgo/Fin.AI", help="HF repo ID")
    parser.add_argument("--model-dir", type=str, default="checkpoints/model", help="Model directory")
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token:
        print("❌ No token provided!")
        print("   Use --token YOUR_TOKEN or set HF_TOKEN environment variable")
        return 1
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1
    
    api = HfApi(token=token)
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        return 1
    
    print(f"📦 Uploading model from {args.model_dir} to {args.repo}")
    
    # Create repo if needed
    try:
        create_repo(args.repo, token=token, exist_ok=True, repo_type="model")
        print(f"✅ Repository {args.repo} ready")
    except Exception as e:
        print(f"⚠️ Repo note: {e}")
    
    # Create model card
    model_card = f"""---
license: mit
tags:
  - pytorch
  - gpt2
  - text-generation
  - fin-ai
---

# Fin.AI

A lightweight, trainable GPT-style language model with automated daily training.

## Model Details

- **Architecture**: GPT-2 style transformer
- **Parameters**: ~10M (tiny preset)
- **Training**: Daily automated training via GitHub Actions
- **Datasets**: Rotating daily (WikiText, TinyStories, CNN, etc.)

## Usage

```python
from huggingface_hub import hf_hub_download

# Download model
hf_hub_download("MeridianAlgo/Fin.AI", "model.pt", local_dir="./model")
hf_hub_download("MeridianAlgo/Fin.AI", "config.json", local_dir="./model")

# Load with Fin.AI
from fin_ai.model import FinAIModel
model = FinAIModel.from_pretrained("./model")
```

## Training Info

- **Last Updated**: {datetime.now().strftime("%Y-%m-%d")}
- **Source**: [GitHub Repository](https://github.com/MeridianAlgo/FinAI)

## License

MIT License
"""
    
    # Save model card
    readme_path = os.path.join(args.model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)
    print("✅ Created model card")
    
    # Upload
    print("⬆️ Uploading to Hugging Face...")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo,
        token=token,
        commit_message=f"Initial upload - {datetime.now().strftime('%Y-%m-%d')}"
    )
    
    print(f"✅ Model uploaded to https://huggingface.co/{args.repo}")
    return 0

if __name__ == "__main__":
    exit(main())
