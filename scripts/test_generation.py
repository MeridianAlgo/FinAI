import torch
from transformers import AutoTokenizer

from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM


def test_gen():
    print("Loading model for generation test...")
    model_path = "./checkpoints_next/model"

    # Load model
    try:
        model = FinAINextForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load from {model_path}, using init: {e}")
        config = FinAINextConfig()
        model = FinAINextForCausalLM(config)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    prompts = [
        "The market broke out of the consolidation phase because",
        "Artificial Intelligence is evolving rapidly. In the future,",
        "def fibonacci(n):",
    ]

    print("\n--- Generation Test ---")
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            # Simple generation
            outputs = model.generate(
                **inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {p}")
        print(f"Output: {text}")


if __name__ == "__main__":
    test_gen()
