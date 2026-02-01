import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM

load_dotenv()


def push():
    print("Seeding Hugging Face repository with initial model...")
    repo_id = "MeridianAlgo/FinAI-Lite"
    model_path = "./checkpoints_next/model"

    # 1. Setup Config (Exactly as in train.py)
    config = FinAINextConfig(
        vocab_size=151665,
        hidden_size=1536,
        num_layers=24,
        liquid_state_dim=384,
        gradient_checkpointing=True,
    )

    # 2. Initialize Model
    print("Initializing model architecture...")
    model = FinAINextForCausalLM(config)

    # 3. Save locally
    print(f"Saving architecture to {model_path}...")
    os.makedirs(model_path, exist_ok=True)
    # Save config and model architecture metadata
    model.config.save_pretrained(model_path)
    
    # We do NOT save the weights here to keep the seed light.
    # We also want to include the tokenizer config if possible
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.save_pretrained(model_path)
    except Exception as e:
        print(f"Warning: Could not save tokenizer to seed: {e}")

    # 4. Copy Model Card
    print("Adding Model Card...")
    import shutil
    if os.path.exists("MODEL_CARD.md"):
        shutil.copy("MODEL_CARD.md", os.path.join(model_path, "README.md"))

    # 5. Push to HF
    print(f"Pushing seed to HF: {repo_id}...")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not found in .env")
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message="Initial FinAI-Next Architecture Seed (Config + Tokenizer)",
        token=hf_token,
    )
    print("Seed successful! HF Repo now contains architecture but no weights yet.")


if __name__ == "__main__":
    push()
