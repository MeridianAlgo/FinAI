"""
Example: Exploring the Model Internals

This script shows how to instantiate the MeridianSMoEConfig and MeridianSMoEForCausalLM
directly from the source code, without relying on Hugging Face AutoClasses.
Useful for debugging or modifying the architecture.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.model.configuration import MeridianSMoEConfig
from meridian.model.modeling import MeridianSMoEForCausalLM


def main():
    print("Creating Meridian.AI Model Configuration...")
    # Default configuration parameters can be overridden
    config = MeridianSMoEConfig(
        vocab_size=151936,  # Qwen2.5 tokenizer vocabulary
        hidden_size=1024,  # Hidden dimension
        num_layers=14,  # Alternating dense/MoE layers
        num_attention_heads=12,
        num_key_value_heads=4,  # Grouped Query Attention
        num_experts=8,  # Experts per MoE layer
        num_experts_per_token=2,  # Active experts per token
        use_numeracy_encoding=True,  # Special numeric embeddings
    )

    print("\nConfiguration details:")
    print(f" - Hidden size: {config.hidden_size}")
    print(f" - Layers: {config.num_layers}")
    print(f" - MoE Experts: {config.num_experts} (Active: {config.num_experts_per_token})")

    print("\nInstantiating model from scratch (random weights)...")
    model = MeridianSMoEForCausalLM(config)

    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters: {total_params / 1e6:.1f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.1f} M")

    print("\nModel Architecture Summary:")
    # Print the first few layers to see the alternating structure
    # layer.is_moe is True for MoE layers, False for dense layers
    for i in range(min(4, len(model.model.layers))):
        layer = model.model.layers[i]
        layer_type = "MoE" if layer.is_moe else "Dense"
        print(f"Layer {i} ({layer_type}): is_moe={layer.is_moe}")


if __name__ == "__main__":
    main()
