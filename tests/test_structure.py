#!/usr/bin/env python3
"""
Test script to verify the new modular structure
"""
import os
import sys

def test_structure():
    """Test that all required files and directories exist"""
    print("Testing FinAI structure...\n")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'MIGRATION_GUIDE.md',
        'ARCHITECTURE.md',
        'PROJECT_SUMMARY.md',
        'TRAINING_DATA_EXAMPLE.txt',
        'src/__init__.py',
        'src/config.py',
        'src/data/__init__.py',
        'src/data/tokenizer.py',
        'src/data/dataset_loader.py',
        'src/models/__init__.py',
        'src/models/language_model.py',
        'src/models/text_generator.py',
        'src/core/__init__.py',
        'src/core/context.py',
        'src/core/finai.py',
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    print()
    
    if missing:
        print(f"❌ {len(missing)} files missing!")
        return False
    else:
        print("✅ All files present!")
        return True

def test_imports():
    """Test that all modules can be imported"""
    print("\nTesting imports...\n")
    
    try:
        from src.config import Config
        print("✓ src.config.Config")
        
        from src.data.tokenizer import Tokenizer
        print("✓ src.data.tokenizer.Tokenizer")
        
        from src.data.dataset_loader import DatasetLoader
        print("✓ src.data.dataset_loader.DatasetLoader")
        
        from src.models.language_model import LanguageModel
        print("✓ src.models.language_model.LanguageModel")
        
        from src.models.text_generator import TextGenerator
        print("✓ src.models.text_generator.TextGenerator")
        
        from src.core.context import ConversationContext
        print("✓ src.core.context.ConversationContext")
        
        from src.core.finai import FinAI
        print("✓ src.core.finai.FinAI")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\nTesting configuration...\n")
    
    try:
        from src.config import Config
        config = Config()
        
        print(f"✓ VOCAB_SIZE: {config.VOCAB_SIZE}")
        print(f"✓ MAX_SEQUENCE_LENGTH: {config.MAX_SEQUENCE_LENGTH}")
        print(f"✓ TEMPERATURE: {config.TEMPERATURE}")
        print(f"✓ HIDDEN_DIM: {config.HIDDEN_DIM}")
        
        print("\n✅ Configuration loaded!")
        return True
    except Exception as e:
        print(f"\n❌ Configuration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("FinAI Structure Test")
    print("=" * 70)
    print()
    
    results = []
    
    # Test structure
    results.append(test_structure())
    
    # Test imports
    results.append(test_imports())
    
    # Test config
    results.append(test_config())
    
    # Summary
    print("\n" + "=" * 70)
    if all(results):
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Train the model: python main.py train TRAINING_DATA_EXAMPLE.txt")
        print("2. Run the chat: python main.py")
    else:
        print("❌ Some tests failed!")
        return 1
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
