#!/usr/bin/env python3
"""Force upload the new model to Hugging Face, replacing everything."""

import os
from huggingface_hub import HfApi, create_repo
from datetime import datetime
from pathlib import Path

# Load .env file
def load_env():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
        print("✅ Loaded .env file")
    else:
        print("⚠️  No .env file found")

def main():
    print("🚀 Force Uploading NEW Model to Hugging Face")
    print("=" * 60)
    
    # Load environment variables
    load_env()
    
    # Get token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN not found in .env file")
        print("\nCreate a .env file with:")
        print('HF_TOKEN=your_token_here')
        print("\nOr set it with: export HF_TOKEN=your_token_here")
        return
    
    print(f"✅ Found HF_TOKEN: {token[:10]}...")
    
    repo_id = "MeridianAlgo/Fin.AI"
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create/ensure repo exists
    print(f"\n📦 Ensuring repository {repo_id} exists...")
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"✅ Repository ready")
    except Exception as e:
        print(f"⚠️  Note: {e}")
    
    # Delete all existing files first (fresh start)
    print(f"\n🗑️  Deleting old model files from Hugging Face...")
    try:
        files_to_delete = ["model.pt", "config.json", "README.md", "pytorch_model.bin"]
        for file in files_to_delete:
            try:
                api.delete_file(
                    path_in_repo=file,
                    repo_id=repo_id,
                    token=token,
                    commit_message=f"Delete old {file} for v2.0 fresh start"
                )
                print(f"   ✓ Deleted: {file}")
            except Exception as e:
                if "404" not in str(e):
                    print(f"   ⚠️  {file}: {str(e)[:50]}")
    except Exception as e:
        print(f"⚠️  Cleanup note: {e}")
    
    # Upload new model
    print(f"\n📤 Uploading NEW model files...")
    print("   This may take a few minutes (121 MB)...")
    
    try:
        api.upload_folder(
            folder_path="checkpoints/model",
            repo_id=repo_id,
            token=token,
            commit_message=f"🔥 v2.0.0 - Fresh model initialization - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )
        print(f"✅ Upload successful!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ NEW MODEL UPLOADED TO HUGGING FACE!")
    print("=" * 60)
    print(f"\n🔗 View at: https://huggingface.co/{repo_id}")
    print("\nThe old model has been completely replaced.")
    print("GitHub Actions will now use this new model for training.")
    print("\n📊 Training will start automatically every 1h 10min")
    print("📈 Monitor progress at: https://wandb.ai/meridianalgo-meridianalgo/fin-ai")

if __name__ == "__main__":
    main()
