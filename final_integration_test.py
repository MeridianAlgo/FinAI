#!/usr/bin/env python3
"""Final integration test - simulates complete GitHub Actions workflow"""

import os
import sys

import torch
from transformers import AutoTokenizer

print("🧪 FINAL INTEGRATION TEST - Simulating GitHub Actions\n")
print("=" * 70)

# Step 1: Simulate downloading from HuggingFace
print("\n📥 STEP 1: Download model from HuggingFace")
print("-" * 70)

model_dir = "checkpoints/model"
required_files = ["model.pt", "config.json"]

all_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

if all_exist:
    print("✅ Model files found (simulating HF download):")
    for f in required_files:
        path = os.path.join(model_dir, f)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"   ✓ {f} ({size:.1f} MB)")
else:
    print("❌ Model files missing! Need to initialize first.")
    print("   Run: python scripts/init_v2_model.py")
    sys.exit(1)

# Step 2: Load model
print("\n🤖 STEP 2: Load v2 model")
print("-" * 70)

from fin_ai.model import FinAIModel

try:
    model = FinAIModel.from_pretrained(model_dir)
    print("✅ Model loaded successfully!")
    print("   Architecture: v2 (GQA, SwiGLU, RMSNorm, RoPE)")
    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Layers: {model.config.n_layers}")
    print(f"   Heads: {model.config.n_heads} (KV: {model.config.n_kv_heads})")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Step 3: Prepare training
print("\n📚 STEP 3: Prepare training data")
print("-" * 70)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create tiny test dataset
test_texts = [
    "The future of artificial intelligence is bright.",
    "Machine learning models are becoming more efficient.",
    "Deep learning has revolutionized computer vision.",
] * 5

from torch.utils.data import DataLoader, Dataset


class TestDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.encodings.append(
                {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": enc["input_ids"].squeeze(0),
                }
            )

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


dataset = TestDataset(test_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"✅ Dataset ready: {len(dataset)} samples")

# Step 4: Train for a few steps
print("\n🚀 STEP 4: Train model (5 steps)")
print("-" * 70)

from fin_ai.training import FinAITrainer, TrainingConfig

os.environ["WANDB_MODE"] = "disabled"

config = TrainingConfig(
    batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    max_steps=5,
    log_steps=1,
    save_steps=100,
    use_wandb=False,
    output_dir="./test_final_checkpoints",
)

trainer = FinAITrainer(model=model, train_dataloader=dataloader, config=config)

try:
    print("Training...")
    trainer.train()
    print("✅ Training completed!")
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 5: Test generation
print("\n🎯 STEP 5: Test generation")
print("-" * 70)

model.eval()
prompt = "The future of AI"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(
        inputs["input_ids"],
        max_new_tokens=15,
        temperature=0.8,
        top_k=50,
    )

output = tokenizer.decode(generated[0], skip_special_tokens=True)
print("✅ Generation works!")
print(f"   Input:  '{prompt}'")
print(f"   Output: '{output}'")

# Step 6: Verify model can be saved
print("\n💾 STEP 6: Test model saving")
print("-" * 70)

save_dir = "./test_final_checkpoints/model"
try:
    model.save_pretrained(save_dir)
    print(f"✅ Model saved to {save_dir}")

    # Verify files exist
    if os.path.exists(os.path.join(save_dir, "model.pt")):
        print("   ✓ model.pt created")
    if os.path.exists(os.path.join(save_dir, "config.json")):
        print("   ✓ config.json created")
except Exception as e:
    print(f"❌ Save failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅ FINAL INTEGRATION TEST PASSED!")
print("=" * 70)
print("\n🎯 Summary:")
print("   ✓ Model download simulation: PASSED")
print("   ✓ Model loading: PASSED")
print("   ✓ Training pipeline: PASSED")
print("   ✓ Generation: PASSED")
print("   ✓ Model saving: PASSED")
print("\n🚀 Ready for GitHub Actions deployment!")
print("\n📋 What will happen on GitHub Actions:")
print("   1. Download v2 model from HuggingFace ✓")
print("   2. Load model and continue training ✓")
print("   3. Train for 1000 steps ✓")
print("   4. Save checkpoint ✓")
print("   5. Upload to HuggingFace ✓")
print("   6. Repeat every ~85 minutes ✓")
print("\n✨ Everything is ready!")
