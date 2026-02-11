"""Count model parameters by component for analysis."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.model.configuration import MeridianConfig
from meridian.model.modeling import MeridianForCausalLM


def main():
    config = MeridianConfig()
    model = MeridianForCausalLM(config)

    total = 0
    components = {}

    for name, param in model.named_parameters():
        numel = param.numel()
        total += numel

        # Group by component
        parts = name.split(".")
        if len(parts) >= 2:
            component = f"{parts[0]}.{parts[1]}"
        else:
            component = parts[0]

        components[component] = components.get(component, 0) + numel

    print(f"\n{'='*60}")
    print(f"  MeridianFormer Parameter Report")
    print(f"{'='*60}\n")
    print(f"  Total: {total:>15,}")
    print(f"  Total (M): {total / 1e6:>11.1f}M\n")

    # Sort by size
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {comp:<35} {count:>12,} ({pct:>5.1f}%) {bar}")

    # Active params per token (MoE)
    moe_expert_params = 0
    dense_params = 0
    for name, param in model.named_parameters():
        if "experts" in name:
            moe_expert_params += param.numel()
        else:
            dense_params += param.numel()

    active_expert_params = moe_expert_params * config.num_experts_per_token / config.num_experts
    active_total = dense_params + active_expert_params

    print(f"\n  {'='*50}")
    print(f"  Active params per token: {active_total:,.0f} ({active_total / 1e6:.1f}M)")
    print(f"  Speedup ratio: {total / active_total:.2f}x")
    print(f"  {'='*50}\n")


if __name__ == "__main__":
    main()
