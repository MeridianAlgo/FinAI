import os

from huggingface_hub import HfApi, list_repo_files


def clean_hf_repo():
    repo_id = "MeridianAlgo/Fin.AI"
    token = os.environ.get("HF_TOKEN")

    if not token:
        # Try to read from .env
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.strip().split("=", 1)[1]
                        break

    if not token:
        print("Error: HF_TOKEN not found.")
        return

    api = HfApi(token=token)

    print(f"Cleaning repository: {repo_id}")
    try:
        files = list_repo_files(repo_id, token=token, repo_type="model")
        print(f"Found {len(files)} files.")

        files_to_delete = [
            f for f in files if f != ".gitattributes" and f != "README.md"
        ]

        if not files_to_delete:
            print("Repo already clean (or only has immutable files).")
            return

        print(f"Deleting {len(files_to_delete)} files...")

        # Delete in batches or one by one
        # Commit operation for deletion
        operations = []
        for f in files_to_delete:
            api.delete_file(path_in_repo=f, repo_id=repo_id, token=token)
            print(f"Deleted {f}")

        print("Repository cleaned.")

    except Exception as e:
        print(f"Error cleaning repo: {e}")


if __name__ == "__main__":
    clean_hf_repo()
