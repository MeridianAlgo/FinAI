#!/usr/bin/env python3
"""Initialize a fresh new Fin.AI model from scratch."""

import os
import shutil
import torch
import json
from transformers import AutoTokenizer
from fin_ai.model import FinAIModel, FinAIConfig

def main():
    print("🔥 Initializing FRESH Fin.AI Model")
    print("=" * 60)
    
    # Delete old checkpoint if exists
    checkpoint_dir = "checkpoints/model"
    if os.path.exists(checkpoint_dir):
        print(f"🗑️  Deleting old model at {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
    
    # Create fresh directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✨ Created fresh directory: {checkpoint_dir}")
    
    # Load tokenizer
    print("🔤 Loading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model config
    print("⚙️  Creating model configuration (small preset)...")
    config = FinAIConfig.from_preset(
        "small",
        vocab_size=len(tokenizer),
        max_seq_len=512,
    )
    
    # Create fresh model
    print("🤖 Initializing fresh model...")
    model = FinAIModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Layers: {config.n_layers}")
    print(f"   Embedding dim: {config.embed_dim}")
    print(f"   Attention heads: {config.n_heads}")
    
    # Save model
    model_path = os.path.join(checkpoint_dir, "model.pt")
    print(f"💾 Saving model to {model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'step': 0,
        'epoch': 0,
    }, model_path)
    
    # Save config
    config_path = os.path.join(checkpoint_dir, "config.json")
    print(f"💾 Saving config to {config_path}...")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Create version file
    version_path = "checkpoints/version.json"
    print(f"💾 Creating version file at {version_path}...")
    with open(version_path, 'w') as f:
        json.dump({
            "version": "v2.0.0",
            "model_type": "fresh_init",
            "parameters": total_params,
            "architecture": "small",
            "note": "Fresh model initialization - starting training from scratch"
        }, f, indent=2)
    
    # Delete old dataset state to start fresh
    dataset_state = "checkpoints/dataset_state.json"
    if os.path.exists(dataset_state):
        print(f"🗑️  Deleting old dataset state...")
        os.remove(dataset_state)
    
    print("\n" + "=" * 60)
    print("✅ NEW MODEL INITIALIZED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python train.py --max-steps 100  (test locally)")
    print("2. Commit and push to trigger GitHub Actions")
    print("3. Model will be uploaded to Hugging Face automatically")
    print("\nNote: This is a FRESH model starting from random weights.")
    print("Training will take 2-4 weeks for coherent outputs.")

if __name__ == "__main__":
    main()
