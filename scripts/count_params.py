"""Count model parameters by component for analysis.

Loads the current Meridian.AI checkpoint (or Qwen2.5-0.5B base) and reports
parameter counts by layer group.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    model_id = os.getenv("HF_MODEL_ID", "meridianal/FinAI")
    subfolder = os.getenv("SUBFOLDER", "checkpoint")
    print(f"Loading {model_id} ({subfolder})...")

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM

    token = os.getenv("HF_TOKEN") or os.getenv("huggingface_token")

    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            token=token,
            allow_patterns=[f"{subfolder}/*"],
        )
        model_path = os.path.join(local_dir, subfolder)
    except Exception as e:
        print(f"  [WARN] Could not download checkpoint ({e}). Falling back to base model.")
        model_path = "Qwen/Qwen2.5-0.5B"

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)

    total = 0
    components: dict[str, int] = {}

    for name, param in model.named_parameters():
        numel = param.numel()
        total += numel
        parts = name.split(".")
        component = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        components[component] = components.get(component, 0) + numel

    print(f"\n{'=' * 60}")
    print("  Meridian.AI Parameter Report")
    print(f"  Model: {model_path}")
    print(f"{'=' * 60}\n")
    print(f"  Total parameters: {total:>15,}")
    print(f"  Total (M):        {total / 1e6:>11.1f}M\n")

    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total
        print(f"  {comp:<40} {count:>12,}  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
