
import os
import json
import shutil
from huggingface_hub import HfApi, create_repo
from fin_ai.model import FinAIConfig, FinAIForCausalLM

def init_and_push():
    repo_id = "MeridianAlgo/Fin.AI"
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.strip().split("=", 1)[1]
                        break
    
    if not token:
        print("HF_TOKEN not found.")
        return

    print("Initializing v4 model state...")
    
    # Create a fresh config
    config = FinAIConfig(size_preset="micro")
    # Ensure auto_map is set for custom model
    config.auto_map = {
        "AutoConfig": "configuration_finai.FinAIConfig",
        "AutoModelForCausalLM": "modeling_finai.FinAIForCausalLM"
    }
    
    # Create directory for artifacts
    os.makedirs("temp_hf_upload", exist_ok=True)
    
    # Save config
    config.save_pretrained("temp_hf_upload")
    
    # Copy code files
    code_files = [
        ("fin_ai/model/configuration_finai.py", "configuration_finai.py"),
        ("fin_ai/model/modeling_finai.py", "modeling_finai.py"),
        ("fin_ai/model/__init__.py", "__init__.py")
    ]
    
    for src, dst in code_files:
        shutil.copy(src, os.path.join("temp_hf_upload", dst))
        
    # Create a basic README / Model Card
    with open("temp_hf_upload/README.md", "w") as f:
        f.write("---\n")
        f.write(f"library_name: transformers\n")
        f.write(f"tags:\n- fin-ai\n- continuous-training\n- fineweb-edu\n")
        f.write("---\n\n# Fin.AI v4\n\nFreshly initialized model with fixed attention mechanism.")

    # Upload
    api = HfApi(token=token)
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    
    print(f"Uploading initialized model to {repo_id}...")
    api.upload_folder(
        folder_path="temp_hf_upload",
        repo_id=repo_id,
        token=token,
        repo_type="model",
        commit_message="Initialize v4 model structure"
    )
    print("Done.")

    # Cleanup
    shutil.rmtree("temp_hf_upload")

if __name__ == "__main__":
    init_and_push()
