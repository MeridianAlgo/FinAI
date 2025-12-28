#!/usr/bin/env python3
"""Force upload the new model to Hugging Face, replacing everything."""

import os
from huggingface_hub import HfApi, create_repo
from datetime import datetime

def main():
    print("🚀 Force Uploading NEW Model to Hugging Face")
    print("=" * 60)
    
    # Get token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN environment variable not set")
        print("Set it with: export HF_TOKEN=your_token_here")
        return
    
    repo_id = "MeridianAlgo/Fin.AI"
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create/ensure repo exists
    print(f"📦 Ensuring repository {repo_id} exists...")
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"✅ Repository ready")
    except Exception as e:
        print(f"⚠️  Note: {e}")
    
    # Delete all existing files first (fresh start)
    print(f"\n🗑️  Deleting old model files from Hugging Face...")
    try:
        files_to_delete = ["model.pt", "config.json", "README.md"]
        for file in files_to_delete:
            try:
                api.delete_file(
                    path_in_repo=file,
                    repo_id=repo_id,
                    token=token,
                    commit_message=f"Delete old {file} for v2.0 fresh start"
                )
                print(f"   Deleted: {file}")
            except Exception as e:
                print(f"   Skip {file}: {str(e)[:50]}")
    except Exception as e:
        print(f"⚠️  Cleanup note: {e}")
    
    # Upload new model
    print(f"\n📤 Uploading NEW model files...")
    
    try:
        api.upload_folder(
            folder_path="checkpoints/model",
            repo_id=repo_id,
            token=token,
            commit_message=f"🔥 v2.0.0 - Fresh model initialization - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            delete_patterns=["*"],  # Delete everything first
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

if __name__ == "__main__":
    main()
