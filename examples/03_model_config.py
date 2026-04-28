"""
Example: Exploring the Model Internals

This script shows how to instantiate the MeridianConfig and MeridianForCausalLM
directly from the source code, without relying on Hugging Face AutoClasses.
Useful for debugging or modifying the architecture.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.model.configuration import MeridianConfig
from meridian.model.modeling import MeridianForCausalLM

def main():
    print("Creating Meridian.AI Model Configuration...")
    # Default configuration parameters can be overridden
    config = MeridianConfig(
        vocab_size=151936,     # Qwen2.5 tokenizer vocabulary
        hidden_size=1024,      # Hidden dimension
        num_hidden_layers=14,  # Alternating dense/MoE layers
        num_attention_heads=12,
        num_key_value_heads=4, # Grouped Query Attention
        moe_num_experts=8,     # Experts per MoE layer
        moe_top_k=2,           # Active experts per token
        use_numeracy=True,     # Special numeric embeddings
    )
    
    print("\nConfiguration details:")
    print(f" - Hidden size: {config.hidden_size}")
    print(f" - Layers: {config.num_hidden_layers}")
    print(f" - MoE Experts: {config.moe_num_experts} (Active: {config.moe_top_k})")
    
    print("\nInstantiating model from scratch (random weights)...")
    model = MeridianForCausalLM(config)
    
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters: {total_params / 1e6:.1f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.1f} M")
    
    print("\nModel Architecture Summary:")
    # Print the first few layers to see the alternating structure
    print("Layer 0 (Dense):", type(model.model.layers[0].mlp).__name__)
    print("Layer 1 (MoE):", type(model.model.layers[1].mlp).__name__)
    print("Layer 2 (Dense):", type(model.model.layers[2].mlp).__name__)
    print("Layer 3 (MoE):", type(model.model.layers[3].mlp).__name__)

if __name__ == "__main__":
    main()
