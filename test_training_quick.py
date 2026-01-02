#!/usr/bin/env python3
"""Quick training test - simulates GitHub Actions training"""

import os
import sys
import torch
from transformers import AutoTokenizer

from fin_ai.model import FinAIModel, FinAIConfig
from fin_ai.data import load_datasets_from_config, create_dataloader
from fin_ai.training import FinAITrainer, TrainingConfig

print("🧪 Quick Training Test\n")

# Disable wandb for quick test
os.environ["WANDB_MODE"] = "disabled"

# Load configs
print("⚙️  Loading configurations...")
model_config = FinAIConfig.from_preset("tiny")  # Use tiny for speed
model_config.max_seq_len = 128  # Shorter sequences
training_config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-4,
    max_steps=10,
    log_steps=2,
    save_steps=100,  # Don't save during test
    use_wandb=False,
    output_dir="./test_checkpoints",
)

# Load tokenizer
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model_config.vocab_size = len(tokenizer)

# Create a tiny test dataset
print("📚 Creating test dataset...")
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world.",
    "Python is a powerful programming language.",
    "Artificial intelligence will change everything.",
    "Deep learning models are getting smarter.",
] * 20  # Repeat to have enough data

from torch.utils.data import Dataset


class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = []
        for text in texts:
            encoded = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.encodings.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    "labels": encoded["input_ids"].squeeze(0),
                }
            )

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


dataset = SimpleTextDataset(test_texts, tokenizer, max_length=128)
train_dataloader = create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0)

print(f"📊 Dataset: {len(dataset)} samples")

# Create model
print("🤖 Creating model...")
model = FinAIModel(model_config)
total_params = sum(p.numel() for p in model.parameters())
print(f"✨ Model ready: {total_params:,} parameters")

# Create trainer
print("\n🚀 Starting training...")
trainer = FinAITrainer(
    model=model,
    train_dataloader=train_dataloader,
    config=training_config,
)

# Train
try:
    trainer.train()
    print("\n✅ Training completed successfully!")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    sys.exit(1)

# Test generation
print("\n🎯 Testing generation...")
model.eval()
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.generate(
        inputs["input_ids"],
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )

output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"\n📝 Generated text:")
print(f"   Prompt: '{prompt}'")
print(f"   Output: '{output_text}'")

print("\n✅ All tests passed! Training pipeline works correctly.")
print("\n🎯 Summary:")
print(f"   - Model: {total_params:,} parameters")
print(f"   - Training steps: {training_config.max_steps}")
print(f"   - Dataset: {len(dataset)} samples")
print(f"   - Generation: Working ✓")
print("\n🚀 Ready for GitHub Actions deployment!")
