
import torch

from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM


def test_forward():
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
    
    # Try loading local states if they exist
    if torch.os.path.exists("checkpoints/model/pytorch_model.bin"):
        print("Loading local weights...")
        model.load_state_dict(torch.load("checkpoints/model/pytorch_model.bin", map_location="cpu"))
    
    model.eval()
    input_ids = torch.randint(0, 52000, (1, 128))
    labels = input_ids.clone()
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
    
    print(f"Loss: {outputs.loss}")
    if torch.isnan(outputs.loss):
        print("❌ Loss is NaN!")
    else:
        print("✓ Loss is healthy.")
        
    print(f"Logits range: [{outputs.logits.min()}, {outputs.logits.max()}]")

if __name__ == "__main__":
    test_forward()
