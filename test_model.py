#!/usr/bin/env python3
"""Quick test of the downloaded model"""

import torch
import json
from transformers import AutoTokenizer
from fin_ai.model import FinAIModel, FinAIConfig

# Load config and model
print("Loading model...")
with open("downloaded_model/config.json", "r") as f:
    config_dict = json.load(f)

config = FinAIConfig(**config_dict)
model = FinAIModel(config)

# Load weights
checkpoint = torch.load("downloaded_model/model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint)
model.eval()

print(f"✅ Model loaded: {config.num_parameters:,} parameters\n")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test prompt
prompt = "The future of artificial intelligence is"
print(f"Prompt: {prompt}")
print("Generating...\n")

# Tokenize
input_ids = tokenizer.encode(prompt, return_tensors="pt")
attention_mask = torch.ones_like(input_ids)

# Generate
with torch.no_grad():
    for _ in range(50):  # Generate 50 tokens
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        
        # Apply temperature and sample
        temperature = 0.8
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        
        # Stop if EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

# Decode
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(f"Generated:\n{generated_text}")
