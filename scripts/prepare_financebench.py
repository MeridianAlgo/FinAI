#!/usr/bin/env python3
"""
Download and prepare the PatronusAI/financebench dataset for FinAI training
"""
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Install it with: pip install datasets")
    sys.exit(1)

def prepare_financebench():
    """Download and convert financebench dataset to FinAI format"""
    print("Downloading PatronusAI/financebench dataset...")
    print("This may take a few minutes...\n")
    
    try:
        # Load the dataset
        ds = load_dataset("PatronusAI/financebench")
        
        print(f"✓ Dataset loaded successfully!")
        print(f"Available splits: {list(ds.keys())}\n")
        
        # Prepare training data
        output_file = "financebench_training.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            count = 0
            
            # Process each split
            for split_name in ds.keys():
                print(f"Processing split: {split_name}")
                split_data = ds[split_name]
                
                for item in split_data:
                    # Extract question and answer from financebench dataset
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    justification = item.get('justification', '')
                    
                    if question and answer:
                        # Clean and format
                        question = question.strip()
                        answer = answer.strip()
                        
                        # Create a more detailed response by combining answer and justification
                        if justification:
                            # Simplify justification for training
                            justification = justification.strip()
                            # Take first sentence or first 200 chars of justification
                            if '.' in justification[:200]:
                                justification = justification[:justification.find('.', 0, 200) + 1]
                            else:
                                justification = justification[:200] + '...'
                            
                            full_answer = f"{answer} {justification}"
                        else:
                            full_answer = answer
                        
                        # Write in FinAI format (lowercase for consistency)
                        f.write(f"user: {question.lower()}\n")
                        f.write(f"assistant: {full_answer.lower()}\n")
                        f.write("\n")
                        count += 1
                
                print(f"  Processed {len(split_data)} items from {split_name}")
        
        print(f"\n✓ Created training file: {output_file}")
        print(f"✓ Total training examples: {count}")
        print(f"\nNext steps:")
        print(f"1. Review the file: {output_file}")
        print(f"2. Train the model: python main.py train {output_file}")
        print(f"3. Chat with FinAI: python main.py")
        
        return output_file
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify dataset name: PatronusAI/financebench")
        print("3. Try: pip install --upgrade datasets")
        return None

def inspect_dataset():
    """Inspect the dataset structure"""
    print("Inspecting dataset structure...\n")
    
    try:
        ds = load_dataset("PatronusAI/financebench")
        
        # Show structure
        for split_name in ds.keys():
            print(f"Split: {split_name}")
            split_data = ds[split_name]
            print(f"  Size: {len(split_data)}")
            
            if len(split_data) > 0:
                print(f"  Fields: {list(split_data[0].keys())}")
                print(f"  Sample:")
                for key, value in split_data[0].items():
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"    {key}: {value_str}")
            print()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_dataset()
    else:
        prepare_financebench()
