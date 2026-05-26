import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_id = os.getenv("HF_MODEL_ID", "meridianal/FinAI")
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "./checkpoint")
    print(f"[INFO] Download/load: {model_id}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "hidden_act") and config.hidden_act == "swiglu":
        config.hidden_act = "silu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )

    # Save model + tokenizer to checkpoint path
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"[INFO] Saving model + tokenizer to {checkpoint_path}...")
    try:
        tokenizer.save_pretrained(checkpoint_path)
    except Exception as e:
        print(f"[WARN] tokenizer.save_pretrained failed: {e}")
    try:
        model.save_pretrained(checkpoint_path, safe_serialization=True)
    except Exception as e:
        print(f"[WARN] model.save_pretrained failed: {e}")

    print("[OK] Download and save complete.")


if __name__ == "__main__":
    main()
