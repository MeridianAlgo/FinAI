import os

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM


def seed_repo():
    token = os.getenv("HF_TOKEN")
    repo_id = "MeridianAlgo/FinAI-Lite"
    api = HfApi(token=token)

    print(f"Nuking and Seeding {repo_id}...")

    # 1. Nuke existing files if repo exists
    try:
        files = api.list_repo_files(repo_id)
        for file in files:
            if file != ".gitattributes":
                print(f"  - Deleting {file}")
                api.delete_file(path_in_repo=file, repo_id=repo_id)
    except Exception as e:
        print(f"Repo might not exist or error: {e}")
        api.create_repo(repo_id=repo_id, exist_ok=True)

    # 2. Re-initialize Model & Config
    config = FinAINextConfig(
        vocab_size=151665,
        hidden_size=1536,
        num_layers=24,
        liquid_state_dim=384,
        gradient_checkpointing=True,
        tie_word_embeddings=True,
    )
    model = FinAINextForCausalLM(config)

    # 3. Save locally temp
    temp_dir = "./temp_seed"
    os.makedirs(temp_dir, exist_ok=True)
    model.save_pretrained(temp_dir, safe_serialization=True)

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer.save_pretrained(temp_dir)

    # 5. Upload everything
    print("Uploading seed files...")
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        commit_message="Nuke and Seed: Fresh Start",
    )

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)
    print("Seeding complete.")


if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN not found in env")
    else:
        seed_repo()
