"""Update model statistics in README and documentation"""
import os
import sys
import re

# Add project root to path
sys.path.append(os.getcwd())

def count_model_parameters():
    """Calculate the current model's parameter count"""
    try:
        from src.core.finai import FinAI
        from src.config import Config
        import torch
        
        config = Config()
        
        # Check if model exists
        if not os.path.exists(config.LANGUAGE_MODEL_PATH):
            print("No trained model found - using config defaults")
            # Calculate theoretical parameter count based on config
            vocab_size = 50257  # Default vocab size
            n_embd = config.N_EMBD
            n_layer = config.N_LAYER
            n_head = config.N_HEAD
            block_size = config.BLOCK_SIZE
            
            # Approximate parameter calculation
            # Token embeddings: vocab_size * n_embd
            # Each transformer block has:
            #   - Attention: 3 * n_embd * n_embd (QKV) + n_embd * n_embd (proj)
            #   - SwiGLU MLP: ~8/3 expansion factor
            #   - LayerNorms: minimal
            
            embedding_params = vocab_size * n_embd
            attention_params_per_layer = 4 * n_embd * n_embd
            
            # SwiGLU has 3 linear layers with hidden_dim = int(8 * n_embd / 3) rounded to 256
            hidden_dim = ((int(8 * n_embd / 3) + 255) // 256) * 256
            mlp_params_per_layer = n_embd * hidden_dim + hidden_dim * n_embd + n_embd * hidden_dim
            
            layer_params = (attention_params_per_layer + mlp_params_per_layer) * n_layer
            ln_params = 2 * n_embd * (n_layer + 1)  # LayerNorms
            
            total_params = embedding_params + layer_params + ln_params
            
            return total_params
        else:
            # Load actual model and count parameters
            print("Loading trained model to count parameters...")
            from src.models.language_model_pytorch import LanguageModel
            model, _ = LanguageModel.load(config.LANGUAGE_MODEL_PATH, use_gpu=False)
            total_params = sum(p.numel() for p in model.parameters())
            return total_params
    except Exception as e:
        print(f"Error counting parameters: {e}")
        # Fallback to config-based estimate
        return None

def format_param_count(count):
    """Format parameter count in human-readable format"""
    if count is None:
        return "~14M"  # Default fallback
    
    if count >= 1e9:
        return f"{count/1e9:.1f}B"
    elif count >= 1e6:
        return f"{count/1e6:.1f}M"
    elif count >= 1e3:
        return f"{count/1e3:.1f}K"
    else:
        return str(count)

def count_trained_datasets():
    """Count unique datasets trained on"""
    csv_file = "trained_datasets.csv"
    if not os.path.exists(csv_file):
        return 0
    
    try:
        import csv
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            datasets = set()
            for row in reader:
                if row['name']:
                    datasets.add(row['name'])
            return len(datasets)
    except Exception as e:
        print(f"Error counting datasets: {e}")
        return 0

def update_readme(param_count_str, dataset_count):
    """Update README.md with dynamic parameter count and dataset badges"""
    readme_path = "README.md"
    
    if not os.path.exists(readme_path):
        print(f"README.md not found at {readme_path}")
        return
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Model Size Badge
    size_badge = f"![Model Size](https://img.shields.io/badge/Model%20Size-{param_count_str.replace(' ', '%20')}-brightgreen?style=for-the-badge)"
    
    if "![Model Size]" in content:
        content = re.sub(r'!\[Model Size\]\([^)]+\)', size_badge, content)
    else:
        content = re.sub(r'(!\[Status\][^\n]+)', r'\1\n' + size_badge, content)

    # 2. Datasets Learned Badge
    ds_badge = f"![Datasets Learned](https://img.shields.io/badge/Datasets%20Learned-{dataset_count}-blue?style=for-the-badge)"
    
    if "![Datasets Learned]" in content:
        content = re.sub(r'!\[Datasets Learned\]\([^)]+\)', ds_badge, content)
    else:
        # Add after Model Size badge
        if "![Model Size]" in content:
             content = re.sub(r'(!\[Model Size\][^\n]+)', r'\1\n' + ds_badge, content)
        else:
             # Fallback if model size badge missing for some reason
             content = re.sub(r'(!\[Status\][^\n]+)', r'\1\n' + ds_badge, content)

    print(f"Updated badges: Size={param_count_str}, Datasets={dataset_count}")
    
    # Remove any static parameter mentions (e.g., "~14M parameters", "14M param", etc.)
    # But keep the dynamic features section
    patterns_to_remove = [
        r'\*\*Smart & Efficient\*\*: ~\d+[KMB]\s+parameters optimized for financial reasoning\.',
        r'~\d+[KMB]\s+parameters optimized for financial reasoning',
        r'~\d+[KMB]\s+parameter architecture',
        r'\d+[KMB]\s+param model',
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # Clean up the Smart & Efficient bullet point to be dynamic
    content = re.sub(
        r'\*   \*\*Smart & Efficient\*\*:[^\n]*',
        '*   **Smart & Efficient**: Optimized architecture for financial reasoning.',
        content
    )
    
    # Write back
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Updated README.md with model size: {param_count_str} and datasets: {dataset_count}")

def update_docs():
    """Remove static parameter counts from all documentation"""
    docs = ['CONTRIBUTING.md', 'SECURITY.md']
    
    for doc in docs:
        if not os.path.exists(doc):
            continue
        
        with open(doc, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove specific parameter count mentions
        patterns_to_remove = [
            r'~?\d+[KMB]\s+parameters?',
            r'\d+[KMB]\s+param',
        ]
        
        modified = False
        for pattern in patterns_to_remove:
            new_content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            if new_content != content:
                content = new_content
                modified = True
        
        if modified:
            with open(doc, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Updated {doc}")

def main():
    print("="*80)
    print("FinAI Model Statistics Updater")
    print("="*80)
    
    # Count parameters
    print("\nCounting model parameters...")
    param_count = count_model_parameters()
    param_count_str = format_param_count(param_count)
    
    if param_count:
        print(f"Model has {param_count:,} parameters ({param_count_str})")
    else:
        print(f"Using default estimate: {param_count_str}")
    
    # Count datasets
    print("\nCounting trained datasets...")
    dataset_count = count_trained_datasets()
    print(f"Model has trained on {dataset_count} unique datasets")

    # Update README
    print("\nUpdating README.md...")
    update_readme(param_count_str, dataset_count)
    
    # Update other docs
    print("\nUpdating documentation files...")
    update_docs()
    
    print("\n" + "="*80)
    print("✓ Model statistics updated successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
