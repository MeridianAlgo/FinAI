import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def nuke_and_reset():
    repo_id = "MeridianAlgo/FinAI-Lite"
    token = os.getenv("HF_TOKEN")

    if not token:
        print("Error: HF_TOKEN not found in environment variables.")
        return

    api = HfApi()

    print(f"Deleting repository {repo_id}...")
    try:
        api.delete_repo(repo_id=repo_id, token=token, repo_type="model")
        print("Repository deleted successfully.")
    except Exception as e:
        print(f"Warning during deletion (repo might not exist): {e}")

    print(f"Re-creating empty repository {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")
        print("Repository created/reset successfully. It is now empty.")
    except Exception as e:
        print(f"Error creating repository: {e}")


if __name__ == "__main__":
    nuke_and_reset()
