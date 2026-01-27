"""
Manage HuggingFace repositories - list, clean, and delete old models
"""

import os

from huggingface_hub import HfApi, list_repo_files


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


def list_all_repos():
    """List all repositories under your account"""
    token = get_hf_token()
    if not token:
        print("Error: HF_TOKEN not found.")
        return []

    api = HfApi(token=token)

    print("\n=== Your HuggingFace Repositories ===")
    try:
        repos = api.list_models(author="MeridianAlgo", token=token)
        repo_list = []
        for repo in repos:
            print(f"  - {repo.id}")
            repo_list.append(repo.id)
        return repo_list
    except Exception as e:
        print(f"Error listing repos: {e}")
        return []


def list_repo_contents(repo_id):
    """List all files in a specific repository"""
    token = get_hf_token()
    if not token:
        print("Error: HF_TOKEN not found.")
        return

    print(f"\n=== Contents of {repo_id} ===")
    try:
        files = list_repo_files(repo_id, token=token, repo_type="model")
        for f in files:
            print(f"  - {f}")
        print(f"Total: {len(files)} files")
    except Exception as e:
        print(f"Error listing files: {e}")


def clean_repo(repo_id, keep_files=None):
    """Clean a repository, keeping only specified files"""
    if keep_files is None:
        keep_files = [".gitattributes", "README.md"]

    token = get_hf_token()
    if not token:
        print("Error: HF_TOKEN not found.")
        return

    api = HfApi(token=token)

    print(f"\n=== Cleaning {repo_id} ===")
    try:
        files = list_repo_files(repo_id, token=token, repo_type="model")
        print(f"Found {len(files)} files.")

        files_to_delete = [f for f in files if f not in keep_files]

        if not files_to_delete:
            print("Repo already clean (or only has protected files).")
            return

        print(f"\nWill delete {len(files_to_delete)} files:")
        for f in files_to_delete:
            print(f"  - {f}")

        confirm = input("\nProceed with deletion? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            return

        for f in files_to_delete:
            try:
                api.delete_file(path_in_repo=f, repo_id=repo_id, token=token)
                print(f"✓ Deleted {f}")
            except Exception as e:
                print(f"✗ Failed to delete {f}: {e}")

        print("\nRepository cleaned.")
    except Exception as e:
        print(f"Error cleaning repo: {e}")


def delete_repo(repo_id):
    """Delete an entire repository"""
    token = get_hf_token()
    if not token:
        print("Error: HF_TOKEN not found.")
        return

    api = HfApi(token=token)

    print(f"\n=== Deleting {repo_id} ===")
    print("⚠️  WARNING: This will permanently delete the entire repository!")
    confirm = input(f"Type the repo name '{repo_id}' to confirm: ")

    if confirm != repo_id:
        print("Cancelled.")
        return

    try:
        api.delete_repo(repo_id=repo_id, token=token, repo_type="model")
        print(f"✓ Repository {repo_id} deleted successfully.")
    except Exception as e:
        print(f"Error deleting repo: {e}")


def main():
    print("HuggingFace Repository Manager")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("1. List all your repositories")
        print("2. List contents of a specific repo")
        print("3. Clean a repository (remove old files)")
        print("4. Delete an entire repository")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            list_all_repos()

        elif choice == "2":
            repo_id = input(
                "Enter repo ID (e.g., MeridianAlgo/FinAI-Lite): "
            ).strip()
            list_repo_contents(repo_id)

        elif choice == "3":
            repo_id = input("Enter repo ID to clean: ").strip()
            list_repo_contents(repo_id)
            clean_repo(repo_id)

        elif choice == "4":
            repo_id = input("Enter repo ID to DELETE: ").strip()
            delete_repo(repo_id)

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
