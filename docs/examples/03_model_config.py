"""
Example: Exploring the Model Internals

This script shows how to instantiate the MeridianConfig and MeridianForCausalLM
directly from source, without relying on Hugging Face AutoClasses.
Useful for debugging or modifying the architecture.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.model.configuration import MeridianConfig
from meridian.model.modeling import MeridianForCausalLM


def main():
    print("Creating Meridian.AI Model Configuration...")
    config = MeridianConfig(
        vocab_size=151_665,  # Qwen2.5 tokenizer vocabulary
        hidden_size=768,  # Hidden dimension
        num_layers=14,  # Alternating dense/MoE layers
        num_attention_heads=12,
        num_key_value_heads=4,  # Grouped Query Attention
        num_experts=8,  # Experts per MoE layer
        num_experts_per_token=2,  # Active experts per token (top-k)
        use_numeracy_encoding=True,
    )

    print("\nConfiguration details:")
    print(f"  Hidden size:        {config.hidden_size}")
    print(f"  Layers:             {config.num_layers}")
    print(
        f"  Attention heads:    {config.num_attention_heads} Q / {config.num_key_value_heads} KV (GQA)"
    )
    print(
        f"  MoE Experts:        {config.num_experts} total, top-{config.num_experts_per_token} active"
    )
    print(f"  MoE layer freq:     every {config.moe_layer_frequency} layers")
    print(f"  Context window:     {config.max_position_embeddings} tokens")
    print(f"  Numeracy encoding:  {config.use_numeracy_encoding}")

    print("\nInstantiating model from scratch (random weights)...")
    model = MeridianForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Parameters:")
    print(f"  Total:     {total_params / 1e6:.1f}M")
    print(f"  Unique:    {unique_params / 1e6:.1f}M  (tied embeddings deduplicated)")
    print(f"  Trainable: {trainable_params / 1e6:.1f}M")

    print("\nLayer type alternation (dense vs MoE):")
    for i, layer in enumerate(model.model.layers):
        layer_type = "MoE  " if layer.is_moe else "Dense"
        print(f"  Layer {i:2d}: {layer_type}")


if __name__ == "__main__":
    main()
