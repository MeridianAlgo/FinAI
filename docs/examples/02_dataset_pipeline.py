"""
Example: Inspecting the Dataset Pipeline

Meridian.AI uses a custom streaming pipeline to load data across multiple
datasets without downloading them entirely. This script demonstrates how
to initialize the dataloader and view the curriculum mix.
"""
import sys
import os

# Add the root directory to path so we can import meridian
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from meridian.data.pipeline import FinanceDataPipeline, create_dataloader

def main():
    print("Loading tokenizer...")
    # Use a compatible tokenizer for demonstration
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Optionally, limit to light datasets for a faster demo
    os.environ["USE_LIGHT_DATASETS"] = "1"
    
    print("\nInitializing FinanceDataPipeline...")
    # Create the pipeline with a small block size for the demo
    pipeline = FinanceDataPipeline(
        tokenizer=tokenizer,
        block_size=128,
        skip_items=0,
    )
    
    print("\nDataset Curriculum Weights:")
    for ds in pipeline.datasets:
        print(f" - {ds['name']} (split: {ds['split']}): weight {ds['weight']:.2f}")

    print("\nStreaming first 5 items from the mixed pipeline...")
    stream_iter = pipeline.stream()
    
    for i in range(5):
        try:
            item = next(stream_iter)
            print(f"\n--- Item {i+1} ---")
            print(f"Processed Index: {item.get('processed_idx', 'N/A')}")
            
            # Decode the tokens back to text to see what the model learns
            text = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
            print(f"Text Preview:\n{text[:200]}...")
        except StopIteration:
            print("Stream ended unexpectedly.")
            break

if __name__ == "__main__":
    main()
