from datasets import load_dataset

# Load finance-alpaca (best quality)
print("Loading finance-alpaca...")
ds = load_dataset("gbharti/finance-alpaca")
data = ds['train']

output = "finance_alpaca_clean.txt"
count = 0

with open(output, 'w', encoding='utf-8') as f:
    for item in data:
        if 'instruction' in item and 'output' in item:
            q = str(item['instruction']).strip().lower()
            a = str(item['output']).strip().lower()
            
            if len(q) > 20 and len(a) > 20 and len(a) < 500:
                f.write(f"user: {q}\n")
                f.write(f"assistant: {a}\n\n")
                count += 1
                
                if count >= 5000:
                    break

print(f"Created {count} examples in {output}")
print(f"Train with: python main.py train {output}")
