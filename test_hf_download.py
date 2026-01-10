
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

def test_hf_model(model_name="gpt2"):
    print(f"⬇️  Downloading {model_name} from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"✅ Successfully downloaded {model_name}")
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return

    print("🧪 Testing model generation...")
    input_text = "The future of artificial intelligence in finance is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_length=50, 
            num_return_sequences=1, 
            do_sample=True,
            top_k=50
        )
    end_time = time.time()
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n🔮 Input: {input_text}")
    print(f"🤖 Output: {generated_text}")
    print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
    
    model_size = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {model_size:,}")

if __name__ == "__main__":
    test_hf_model()
