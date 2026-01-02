#!/usr/bin/env python3
"""Test downloading and loading model from Hugging Face format"""

import os
import sys
import torch
from transformers import AutoTokenizer

print("🧪 Testing Hugging Face Model Download & Load\n")

# Simulate what GitHub Actions does
print("📥 Simulating GitHub Actions download process...")

# Check if model exists locally
model_dir = "checkpoints/model"
if os.path.exists(os.path.join(model_dir, "model.pt")) and os.path.exists(
    os.path.join(model_dir, "config.json")
):
    print(f"✓ Found model files in {model_dir}")
    print(
        f"  - model.pt: {os.path.getsize(os.path.join(model_dir, 'model.pt')) / 1024 / 1024:.1f} MB"
    )
    print(f"  - config.json: exists")

    # Try to load the model
    print("\n🤖 Loading model...")
    from fin_ai.model import FinAIModel

    try:
        model = FinAIModel.from_pretrained(model_dir)
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {model.count_parameters():,}")
        print(f"   Layers: {model.config.n_layers}")
        print(f"   Heads: {model.config.n_heads}")
        print(f"   KV Heads: {model.config.n_kv_heads}")
        print(f"   Embed dim: {model.config.embed_dim}")

        # Test forward pass
        print("\n🔬 Testing forward pass...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        test_input = tokenizer("Hello world", return_tensors="pt")

        model.eval()
        with torch.no_grad():
            outputs = model(test_input["input_ids"])

        print(f"✅ Forward pass works!")
        print(f"   Output shape: {outputs['logits'].shape}")

        # Test generation
        print("\n🎯 Testing generation...")
        generated = model.generate(
            test_input["input_ids"],
            max_new_tokens=10,
            temperature=0.8,
        )
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✅ Generation works!")
        print(f"   Input: 'Hello world'")
        print(f"   Output: '{output_text}'")

        print("\n✅ All tests passed!")
        print("\n🎯 Summary:")
        print("   ✓ Model files exist")
        print("   ✓ Model loads correctly")
        print("   ✓ Forward pass works")
        print("   ✓ Generation works")
        print("\n🚀 Ready for GitHub Actions training!")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
else:
    print(f"⚠️  Model files not found in {model_dir}")
    print("   This is expected if you haven't uploaded to HuggingFace yet")
    print("   Run: python scripts/init_v2_model.py")
    sys.exit(0)
