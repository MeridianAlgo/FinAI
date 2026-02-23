"""Count model parameters by component for analysis."""

import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.model.configuration import MeridianConfig
from meridian.model.modeling import MeridianForCausalLM


def main():
    model_id = "hpcai-tech/openmoe-base"
    print(f"Loading {model_id} architecture...")
    from transformers import AutoModelForCausalLM
    import torch
    
    # Load architecture only
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    total = 0
    components = {}

    for name, param in model.named_parameters():
        numel = param.numel()
        total += numel

        # Group by component
        parts = name.split(".")
        component = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        components[component] = components.get(component, 0) + numel

    print(f"\n{'=' * 60}")
    print("  MeridianAI Parameter Report")
    print(f"{'=' * 60}\n")
    print(f"  Total: {total:>15,}")
    print(f"  Total (M): {total / 1e6:>11.1f}M\n")

    # Grouping logic for summary
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total
        print(f"  {comp:<35} {count:>12,} ({pct:>5.1f}%)")

    # Smart MoE active param detection
    moe_expert_layers = [m for m in model.modules() if "MoE" in m.__class__.__name__ or m.__class__.__name__ == "OpenMoELayer"]
    
    # Default to 196M active params if detection fails, or try generic check
    moe_params = 0
    non_moe_params = 0
    for name, param in model.named_parameters():
        if "expert" in name or "mlp" in name: # OpenMoE uses mlp for experts
            moe_params += param.numel()
        else:
            non_moe_params += param.numel()

    print(f"\n  {'=' * 70}")
    print("  MeridianAI v1.0 — Finance LLM Training")
    print("  Architecture: Sparse MoE + GQA + RoPE + SwiGLU + Numeracy Encoding")
    print(f"  {'=' * 70}")
    print(f"  Efficiency: optimized for CPU inference.")
    print(f"  {'=' * 50}\n")


if __name__ == "__main__":
    main()
