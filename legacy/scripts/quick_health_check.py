
import os

import torch

from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM


def check_nan(path):
    print(f"Checking model at {path}...")
    if not os.path.exists(path):
        print("Path does not exist!")
        return
    
    try:
        # Load weights but ignore config to just check the bin
        weights = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=False)
        has_nan = False
        for name, param in weights.items():
            if torch.isnan(param).any():
                print(f"❌ Found NaN in {name}")
                has_nan = True
            if torch.isinf(param).any():
                print(f"❌ Found Inf in {name}")
                has_nan = True
        
        if not has_nan:
            print("✓ No NaNs or Infs found in weights.")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_nan("checkpoints/model")
