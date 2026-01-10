
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from fin_ai.model import FinAIModel
import time
import os
import shutil

def test_fin_ai_model():
    repo_id = "MeridianAlgo/Fin.AI"
    local_dir = "./downloaded_model"
    
    print(f"⬇️  Downloading {repo_id} from Hugging Face...")
    
    # Clean up previous download if exists to ensure fresh test
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    try:
        hf_hub_download(repo_id, "model.pt", local_dir=local_dir)
        hf_hub_download(repo_id, "config.json", local_dir=local_dir)
        print(f"✅ Successfully downloaded model files to {local_dir}")
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return

    print("⚙️  Loading model...")
    try:
        model = FinAIModel.from_pretrained(local_dir)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("🔤 Loading tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("🧪 Testing model generation...")
    input_text = "The future of artificial intelligence in finance is"
    print(f"\n🔮 Input: {input_text}")
    
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        # Using the custom generate method from FinAIModel
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=50, 
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
    end_time = time.time()
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"🤖 Output: {generated_text}")
    print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
    
    model_size = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {model_size:,}")

if __name__ == "__main__":
    test_fin_ai_model()
