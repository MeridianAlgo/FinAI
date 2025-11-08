#!/usr/bin/env python3
"""
Test the trained FinAI model on financebench data
"""
from src.core.finai import FinAI

def test_model():
    """Test the model with sample questions"""
    print("=" * 70)
    print("Testing FinAI trained on PatronusAI/financebench")
    print("=" * 70)
    print()
    
    finai = FinAI()
    
    # Load the trained model
    if not finai._load_models():
        print("Error: No trained model found!")
        print("Please train first: python main.py train financebench_training.txt")
        return
    
    print("✓ Model loaded successfully!\n")
    
    # Test questions
    test_questions = [
        "what is capital expenditure",
        "how to calculate operating margin",
        "what is a quick ratio",
        "explain debt securities",
        "what is ppne",
    ]
    
    print("Testing with sample questions:\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. Question: {question}")
        
        # Generate response
        response = finai.generate_response(question)
        
        print(f"   Response: {response[:200]}...")
        print()
    
    print("=" * 70)
    print("Test complete!")
    print("\nTo chat interactively, run: python main.py")
    print("=" * 70)

if __name__ == "__main__":
    test_model()
