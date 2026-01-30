
from huggingface_hub import HfApi
import os

def check_hf():
    token = os.environ.get("HF_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    
    api = HfApi(token=token)
    repo_id = "MeridianAlgo/FinAI-Lite"
    print(f"Checking repo: {repo_id}")
    
    files = api.list_repo_files(repo_id=repo_id)
    print(f"Files in repo: {files}")
    
    info = api.model_info(repo_id=repo_id)
    for sibling in info.siblings:
        if sibling.rfilename == "pytorch_model.bin":
            # Size in bytes
            # Note: sibling doesn't always have size, use info.siblings directly or api.list_repo_tree
            pass
            
    # Better way to get sizes
    tree = api.list_repo_tree(repo_id=repo_id)
    for item in tree:
        if item.path == "pytorch_model.bin":
            size_gb = item.size / (1024**3)
            print(f"pytorch_model.bin size: {size_gb:.2f} GB")
        if item.path == "config.json":
            print(f"config.json found")

if __name__ == "__main__":
    check_hf()
