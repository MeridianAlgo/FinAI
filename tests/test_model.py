#!/usr/bin/env python3
"""
Test Fin.AI model to ensure it's working correctly

Usage:
    python test_model.py
"""

import torch
from transformers import AutoTokenizer

print("🧪 Testing Fin.AI Model\n")

# Test 1: Import modules
print("1️⃣ Testing imports...")
try:
    from fin_ai.model import FinAIModel, FinAIConfig
    from fin_ai.data import load_datasets_from_config
    print("✅ Imports successful\n")
except Exception as e:
    print(f"❌ Import failed: {e}\n")
    exit(1)

# Test 2: Create model
print("2️⃣ Testing model creation...")
try:
    config = FinAIConfig.from_preset("tiny")
    model = FinAIModel(config)
    print(f"✅ Model created: {config.num_parameters:,} parameters\n")
except Exception as e:
    print(f"❌ Model creation failed: {e}\n")
    exit(1)

# Test 3: Forward pass
print("3️⃣ Testing forward pass...")
try:
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    output = model(input_ids)
    
    assert "logits" in output
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    print(f"✅ Forward pass successful: {output['logits'].shape}\n")
except Exception as e:
    print(f"❌ Forward pass failed: {e}\n")
    exit(1)

# Test 4: Loss computation
print("4️⃣ Testing loss computation...")
try:
    labels = input_ids.clone()
    output = model(input_ids, labels=labels)
    
    assert "loss" in output
    assert output["loss"].item() > 0
    print(f"✅ Loss computation successful: {output['loss'].item():.4f}\n")
except Exception as e:
    print(f"❌ Loss computation failed: {e}\n")
    exit(1)

# Test 5: Text generation
print("5️⃣ Testing text generation...")
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "Hello world"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    model.eval()
    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=1.0,
        do_sample=True,
    )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"✅ Generation successful!")
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated: '{generated_text}'\n")
except Exception as e:
    print(f"❌ Generation failed: {e}\n")
    exit(1)

# Test 6: Save and load
print("6️⃣ Testing save/load...")
try:
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model")
        
        # Save
        model.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")
        
        # Load
        loaded_model = FinAIModel.from_pretrained(save_path)
        print(f"✅ Model loaded successfully")
        
        # Compare outputs
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            orig_out = model(input_ids)["logits"]
            loaded_out = loaded_model(input_ids)["logits"]
        
        assert torch.allclose(orig_out, loaded_out, atol=1e-5)
        print(f"✅ Outputs match!\n")
except Exception as e:
    print(f"❌ Save/load failed: {e}\n")
    exit(1)

# Test 7: Dataset loading
print("7️⃣ Testing dataset loading...")
try:
    from datetime import datetime
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    today = datetime.now().weekday()
    
    print(f"   Today is {day_names[today]}")
    
    # This will load today's dataset
    dataset, new_offset = load_datasets_from_config(
        "config/datasets.yaml",
        tokenizer=tokenizer,
        max_seq_len=512,
        max_samples=100,  # Small sample for testing
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} sequences (offset: {new_offset})\n")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}\n")
    exit(1)

print("=" * 50)
print("🎉 All tests passed!")
print("=" * 50)
