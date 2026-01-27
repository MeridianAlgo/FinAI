import os

import yaml
from huggingface_hub import create_repo, upload_folder
from transformers import AutoTokenizer

from fin_ai.model.configuration_finai import FinAIConfig
from fin_ai.model.modeling_finai import FinAIForCausalLM


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    if k.strip() == "HF_TOKEN":
                        return v.strip().strip('"').strip("'")
        except Exception:
            pass
    return token


def main():
    token = get_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment or .env file")

    # Load config
    print("Loading configuration...")
    with open("config/model_config.yaml", "r") as f:
        full_config = yaml.safe_load(f)

    config_dict = full_config.get("model", {})
    repo_id = full_config.get("training", {}).get(
        "hf_repo_id", "MeridianAlgo/FinAI-Lite"
    )

    # Initialize Tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    special_tokens = ["<TICKER>", "<ACCOUNTING>", "<SEC_FILING>", "<ARXIV_FIN>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Update config with vocab size
    config = FinAIConfig(**config_dict)
    config.vocab_size = len(tokenizer)

    # Initialize Model with random weights
    print(
        f"Initializing model with random weights (~{config.hidden_size} hidden size)..."
    )
    model = FinAIForCausalLM(config)
    print(f"Model parameters: {model.num_parameters():,}")

    # Save locally first
    model_path = "checkpoints/random_model"
    os.makedirs(model_path, exist_ok=True)
    # safe_serialization=False required for tied weights in some versions
    model.save_pretrained(model_path, safe_serialization=False)
    tokenizer.save_pretrained(model_path)

    # Push to Hub
    print(f"Pushing to Hugging Face: {repo_id}")
    try:
        create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            token=token,
            commit_message="Initial random weights with novel architecture (Mamba-2, MLA, MoE, MTP)",
        )
        print("Successfully pushed to Hugging Face!")
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")


if __name__ == "__main__":
    main()
