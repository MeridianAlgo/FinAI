import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load the tokenizer
    # The model uses the Qwen2.5-0.5B tokenizer
    repo_id = "meridianal/FinAI"
    print(f"Loading tokenizer from {repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="checkpoint")

    # Load the Meridian.AI MoE model
    print(f"Loading model from {repo_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder="checkpoint",
        trust_remote_code=True, # Required because it's a custom architecture
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Define a prompt using the instruction/response format
    prompt = """### Instruction:
Explain the concept of 'Elastic Weight Consolidation' in continual learning and why it might be useful for a finance model.

### Response:
"""

    print("\nGenerating response...")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and print the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    main()
