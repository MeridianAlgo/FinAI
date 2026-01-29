
import os
import torch
import torch.nn as nn
from fin_ai.model.modeling_finai import FinAIForCausalLM
from fin_ai.model.configuration_finai import FinAIConfig
from huggingface_hub import HfApi, create_repo

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_700m_model():
    # Target: ~700M parameters
    # Embedding: 52000 * 1024 = 53M
    # Layers: 24
    # Each layer:
    #   MLA: q_latent (1024 * 64) + q_heads (64 * 1024) + kv_latent (1024 * 64) + kv_heads (64 * 2048) + o_proj (1024 * 1024)
    #        = 65k + 65k + 65k + 131k + 1M = ~1.3M
    #   MoE: shared (1024 * 2048 * 2) + routed (8 * 1024 * 2048 * 2)
    #        = 4M + 32M = 36M
    # Total per layer: ~37M
    # Total for layers: 24 * 37M = 888M
    # Plus embeddings 53M + heads 53M = 994M.
    
    # Let's reduce experts or layers.
    # Layers 16, Experts 8 -> 16 * 37M = 592M + 106M = 698M. PERFECT!
    
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
    return FinAIForCausalLM(config)

def create_700_tiny_model():
    # Target: ~700 parameters
    # Embedding: 64 * 4 = 256
    # Heads: 256
    # Layer:
    #   MLA: q_latent (4*4) + q_heads (4*4) + kv_latent (4*4) + kv_heads (4*8) + o_proj (4*4) = 16+16+16+32+16 = 96
    #   MLP (no moe): (4*8) + (8*4) = 64
    # Total: 256 + 256 + 96 + 64 = 672. Very close to 700.
    
    config = FinAIConfig(
        vocab_size=64,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=8,
        use_moe=False,
        mla_latent_rank=4,
        max_position_embeddings=128
    )
    return FinAIForCausalLM(config)

def main():
    print("Initializing 700 Million Parameter model (Ultra-Lite)...")
    model = create_700m_model()
    params = count_parameters(model)
    print(f"Model initialized with {params:,} parameters.")
    
    repo_id = "MeridianAlgo/FinAI-Lite"
    print(f"Target Repository: {repo_id}")
    
    # Save locally to a specific folder
    save_path = "checkpoints/model"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path, safe_serialization=False)
    
    # Also save a copy of the config to the root config dir if needed
    print(f"Model saved to {save_path}")
    
    # Push to HF
    token = os.environ.get("HF_TOKEN")
    if not token:
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    
    if token:
        print("Pushing to Hugging Face...")
        api = HfApi(token=token)
        try:
            create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
            api.upload_folder(
                folder_path=save_path,
                repo_id=repo_id,
                commit_message="Initial 700M random model"
            )
            print(f"Successfully pushed to https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"Error during push: {e}")
    else:
        print("HF_TOKEN not found. Skipping push.")

if __name__ == "__main__":
    main()
