#!/usr/bin/env python3
"""
Test DirectML GPU functionality
"""
import torch
import torch_directml

print("Testing DirectML GPU...")

try:
# Create device
device = torch_directml.device()
print(f"Device created: {device}")

# Test basic operations
print("Testing basic tensor operations...")
x = torch.randn(10, 10).to(device)
y = torch.randn(10, 10).to(device)
z = x @ y
print(f"Matrix multiplication successful: {z.shape}")

# Test model forward pass
print("Testing model forward pass...")
model = torch.nn.Linear(10, 5).to(device)
output = model(x)
print(f"Model forward pass successful: {output.shape}")

print(" DirectML GPU is working!")

except Exception as e:
print(f" DirectML GPU failed: {e}")
import traceback
traceback.print_exc()
