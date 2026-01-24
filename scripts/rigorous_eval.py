import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from comet_ml import API
from tqdm import tqdm
import textwrap

# Load environment variables (ensure they are set in the shell or .env is loaded)
# Note: In a real script we might use python-dotenv, but here we assume env vars presence
# or we can read the .env manually if needed, but the user environment should have it.

MODEL_ID = "MeridianAlgo/Fin.AI"
COMET_WORKSPACE = "meridianalgo"
COMET_PROJECT = "fin-ai"

def load_remote_model():
    print(f"Loading model from Hugging Face: {MODEL_ID}...")
    try:
        # README recommends using "gpt2" tokenizer
        print("Loading tokenizer (gpt2)...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        print(f"Loading model ({MODEL_ID})...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            torch_dtype=torch.float32, # Use float32 for CPU safety, or auto for GPU
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

def fetch_comet_data():
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        print("COMET_API_KEY not found. Skipping Comet ML metadata fetch.")
        return

    print("Fetching Comet ML metadata...")
    try:
        api = API(api_key=api_key)
        # Get the latest experiment
        experiments = api.get_experiments(workspace=COMET_WORKSPACE, project_name=COMET_PROJECT)
        if not experiments:
            print("No experiments found in Comet ML.")
            return

        # Sort by end time or creation time to get the latest
        latest_experiment = experiments[0] # API usually returns typically ordered, but let's assume raw list
        
        print(f"Latest Experiment ID: {latest_experiment.id}")
        print(f"Latest Experiment Name: {latest_experiment.name}")
        
        # Try to get metrics
        loss = latest_experiment.get_metrics("train_loss")
        if loss:
            print(f"Latest Train Loss: {loss[-1]['metricValue']}")
        
    except Exception as e:
        print(f"Error fetching Comet data: {e}")

def run_rigorous_tests(model, tokenizer):
    prompts = [
        "Explain the concept of quantum entanglement to a 5-year-old.",
        "Write a python function to calculate the fibonacci sequence using dynamic programming.",
        "What are the main causes of the French Revolution?",
        "Translate 'Hello, how are you?' into French, Spanish, and German.",
        "Analyze the sentiment of this text: 'The market crashed today, causing widespread panic and loss of wealth.'",
        "Who is the president of the United States in 2024?",
        "Define the term 'machine learning' and give three examples of its application."
    ]

    print("\n--- Rigorous Generation Tests ---\n")
    
    generation_config = GenerationConfig(
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    results = []
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                generation_config=generation_config
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response for cleaner output if desired
        response_only = response[len(prompt):]
        
        print(f"Response:\n{textwrap.fill(response_only, width=80)}")
        print("-" * 50)
        results.append({"prompt": prompt, "response": response_only})

    return results

def test_perplexity_on_fineweb(model, tokenizer):
    print("\n--- Testing Perplexity on FineWeb-Edu (Streamed) ---\n")
    try:
        # Load a small slice of FineWeb-Edu
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        iterator = iter(dataset)
        
        total_loss = 0
        num_samples = 5 # Keep it small for quick check
        
        model.eval()
        
        for i in range(num_samples):
            data = next(iterator)
            text = data['text'][:1024] # truncated
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                total_loss += loss.item()
                
            print(f"Sample {i+1} Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / num_samples
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        print(f"\nAverage Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity.item():.4f}")
        
    except Exception as e:
        print(f"Failed to run perplexity test: {e}")

if __name__ == "__main__":
    # Ensure env vars are loaded from .env if possible (rudimentary loader)
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    fetch_comet_data()
    
    model, tokenizer = load_remote_model()
    if model and tokenizer:
        run_rigorous_tests(model, tokenizer)
        test_perplexity_on_fineweb(model, tokenizer)
