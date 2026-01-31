
import torch

from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM


def count_detailed():
    config = FinAIConfig(
        vocab_size=52000,
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2048,
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=2048,
        mla_latent_rank=64,
        max_position_embeddings=4096
    )
    model = FinAIForCausalLM(config)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    
    print("\nBreakdown:")
    print(f"Embedding: {sum(p.numel() for p in model.model.embed_tokens.parameters()):,}")
    print(f"LM Head: {sum(p.numel() for p in model.lm_head.parameters()):,}")
    print(f"MTP Heads: {sum(p.numel() for p in model.mtp_heads.parameters()):,}")
    
    # Check if tied
    if model.lm_head.weight is model.model.embed_tokens.weight:
        print("\n✅ lm_head and embed_tokens are TIED")
    else:
        print("\n❌ lm_head and embed_tokens are NOT TIED")

if __name__ == "__main__":
    count_detailed()
