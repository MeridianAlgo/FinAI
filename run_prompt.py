"""
Interactive prompt for FinAI model
"""
import torch
import os
import sys
from src.models.language_model_pytorch import LanguageModel as PyTorchLanguageModel
from src.data.tokenizer import Tokenizer

def load_model(model_path):
"""Load the trained model and tokenizer"""
# Initialize tokenizer
tokenizer_path = os.path.join('models', 'tokenizer.pkl')

if os.path.exists(tokenizer_path):
tokenizer = Tokenizer.load(tokenizer_path)
else:
print(f"Error: Tokenizer not found at {tokenizer_path}")
sys.exit(1)

# Load model checkpoint
if not os.path.exists(model_path):
print(f"Error: Model not found at {model_path}")
sys.exit(1)

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Initialize model with saved parameters
model = PyTorchLanguageModel(
vocab_size=checkpoint['vocab_size'],
block_size=checkpoint['block_size'],
n_embd=checkpoint['n_embd'],
n_head=checkpoint['n_head'],
n_layer=checkpoint['n_layer'],
dropout=checkpoint['dropout']
)

# Load the model state dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded from {model_path}")

return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
"""Generate text from a prompt"""
# Encode the prompt
tokens = tokenizer.encode(prompt)

# Generate new tokens
for _ in range(max_new_tokens):
# Get the last block_size tokens
idx_cond = tokens[-model.block_size:]

# Convert to tensor on the correct device and add batch dimension
input_tensor = torch.tensor([idx_cond], dtype=torch.long, device=model.device)

# Get predictions
with torch.no_grad():
logits, _ = model(input_tensor)

# Get the logits for the last token
logits = logits[0, -1, :] / temperature

# Apply top-k filtering
if top_k is not None and top_k > 0:
k = min(top_k, logits.size(-1))
v, _ = torch.topk(logits, k)
threshold = v[-1]
logits[logits < threshold] = -float('inf')

# Get probabilities
probs = torch.softmax(logits, dim=-1)

# Sample from the distribution
next_token = torch.multinomial(probs, num_samples=1)
next_token_id = next_token.item()

# Append to sequence
tokens.append(next_token_id)

# Stop if we hit the EOS token
if next_token_id == tokenizer.eos_id:
break

# Decode and return
return tokenizer.decode(tokens)

def interactive_loop():
"""Run interactive prompt loop"""
model_path = os.path.join('models', 'finai_gpt.pt')

print("Loading FinAI model...")
model, tokenizer = load_model(model_path)
print("Model loaded! Type 'exit' to quit.")

while True:
try:
# Get user input
prompt = input("\nEnter your prompt: ")

if prompt.lower() in ['exit', 'quit']:
break

# Generate response
response = generate_text(model, tokenizer, prompt)
print("\nFinAI:", response)

except KeyboardInterrupt:
print("\nExiting...")
break
except EOFError:
print("\nEOF received. Exiting...")
break
except Exception as e:
print(f"Error: {str(e)}")

if __name__ == "__main__":
try:
interactive_loop()
except Exception as e:
print(f"Error: {str(e)}")
sys.exit(1)
