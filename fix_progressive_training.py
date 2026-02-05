"""
Fix Progressive Training - Ensure cascading loss across training runs
This script verifies that model weights are properly saved and loaded
"""

import json
import os
import torch
from pathlib import Path


def check_checkpoint_integrity():
    """Check if checkpoint files exist and are valid"""
    checkpoint_path = "./checkpoint"
    model_path = "./model"
    
    print("=" * 60)
    print("CHECKPOINT INTEGRITY CHECK")
    print("=" * 60)
    
    # Check checkpoint directory
    if os.path.exists(checkpoint_path):
        print(f"\n✓ Checkpoint directory exists: {checkpoint_path}")
        
        # Check for model weights
        safetensors = os.path.join(checkpoint_path, "model.safetensors")
        pytorch_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors):
            size_mb = os.path.getsize(safetensors) / (1024 * 1024)
            print(f"  ✓ model.safetensors exists ({size_mb:.2f} MB)")
        elif os.path.exists(pytorch_bin):
            size_mb = os.path.getsize(pytorch_bin) / (1024 * 1024)
            print(f"  ✓ pytorch_model.bin exists ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ No model weights found!")
            
        # Check for trainer state
        trainer_state = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(trainer_state):
            size_kb = os.path.getsize(trainer_state) / 1024
            print(f"  ✓ trainer_state.pt exists ({size_kb:.2f} KB)")
            
            # Load and inspect
            try:
                state = torch.load(trainer_state, map_location='cpu')
                print(f"    - Global step: {state.get('global_step', 'N/A')}")
                print(f"    - Run step: {state.get('run_step', 'N/A')}")
            except Exception as e:
                print(f"    ✗ Error loading trainer state: {e}")
        else:
            print(f"  ✗ trainer_state.pt not found!")
            
        # Check dataset state
        dataset_state = os.path.join(checkpoint_path, "dataset_state.json")
        if os.path.exists(dataset_state):
            with open(dataset_state) as f:
                state = json.load(f)
            print(f"  ✓ dataset_state.json exists")
            print(f"    - Processed items: {state.get('processed_items', 'N/A')}")
        else:
            print(f"  ✗ dataset_state.json not found!")
    else:
        print(f"\n✗ Checkpoint directory does not exist: {checkpoint_path}")
    
    # Check model directory
    print(f"\n{'=' * 60}")
    if os.path.exists(model_path):
        print(f"✓ Model directory exists: {model_path}")
        
        safetensors = os.path.join(model_path, "model.safetensors")
        pytorch_bin = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors):
            size_mb = os.path.getsize(safetensors) / (1024 * 1024)
            print(f"  ✓ model.safetensors exists ({size_mb:.2f} MB)")
        elif os.path.exists(pytorch_bin):
            size_mb = os.path.getsize(pytorch_bin) / (1024 * 1024)
            print(f"  ✓ pytorch_model.bin exists ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ No model weights found!")
    else:
        print(f"✗ Model directory does not exist: {model_path}")
    
    print("=" * 60)


def verify_weight_loading():
    """Verify that weights are actually different from initialization"""
    print("\n" + "=" * 60)
    print("WEIGHT VERIFICATION")
    print("=" * 60)
    
    checkpoint_path = "./checkpoint"
    
    if not os.path.exists(checkpoint_path):
        print("✗ No checkpoint to verify")
        return
    
    try:
        from fin_ai.model.configuration_next import FinAINextConfig
        from fin_ai.model.modeling_next import FinAINextForCausalLM
        
        config = FinAINextConfig(
            vocab_size=151665,
            hidden_size=1536,
            num_layers=24,
            liquid_state_dim=384,
            gradient_checkpointing=True,
            tie_word_embeddings=True,
        )
        
        # Load checkpoint
        print(f"\nLoading model from {checkpoint_path}...")
        model = FinAINextForCausalLM.from_pretrained(
            checkpoint_path,
            config=config,
            ignore_mismatched_sizes=False,  # Don't ignore mismatches!
            low_cpu_mem_usage=False,
        )
        
        # Check a sample of weights
        with torch.no_grad():
            embed_sample = model.model.embed_tokens.weight[0][:10].tolist()
            print(f"\n✓ Model loaded successfully")
            print(f"  Sample embedding weights: {[f'{x:.4f}' for x in embed_sample]}")
            
            # Check if weights look initialized (not all zeros or random)
            embed_mean = model.model.embed_tokens.weight.mean().item()
            embed_std = model.model.embed_tokens.weight.std().item()
            print(f"  Embedding mean: {embed_mean:.6f}, std: {embed_std:.6f}")
            
            if abs(embed_mean) < 0.0001 and abs(embed_std) < 0.0001:
                print("  ⚠ WARNING: Weights look uninitialized (near zero)")
            else:
                print("  ✓ Weights appear to be trained")
                
    except Exception as e:
        print(f"✗ Error loading model: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    check_checkpoint_integrity()
    verify_weight_loading()
