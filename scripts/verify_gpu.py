#!/usr/bin/env python3
"""Verify GPU availability and configuration"""
import sys

def check_pytorch():
    """Check PyTorch installation and GPU availability"""
    try:
        import torch
        print("=" * 70)
        print("PyTorch GPU Verification")
        print("=" * 70)
        print(f"\nPyTorch version: {torch.__version__}")
        
        # Check CUDA/ROCm
        if torch.cuda.is_available():
            print("\n✓ CUDA/ROCm GPU detected!")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
            print(f"  Device count: {torch.cuda.device_count()}")
            print(f"  CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
            try:
                print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            except:
                pass
            return True
        else:
            print("\n⚠️  No CUDA/ROCm GPU detected")
        
        # Check DirectML
        try:
            import torch_directml
            if torch_directml.is_available():
                print("\n✓ DirectML GPU detected!")
                print(f"  Device: {torch_directml.device()}")
                return True
        except ImportError:
            pass
        
        print("\n⚠️  Using CPU only")
        return False
        
    except ImportError:
        print("\n❌ PyTorch not installed")
        print("Install with: pip install torch")
        return False

def check_numpy():
    """Check NumPy installation"""
    try:
        import numpy as np
        print(f"\nNumPy version: {np.__version__}")
        return True
    except ImportError:
        print("\n❌ NumPy not installed")
        return False

def main():
    """Main verification function"""
    print("\n" + "=" * 70)
    print("FinAI GPU Setup Verification")
    print("=" * 70)
    
    gpu_available = check_pytorch()
    numpy_ok = check_numpy()
    
    print("\n" + "=" * 70)
    if gpu_available:
        print("✓ GPU acceleration is AVAILABLE")
        print("\nYou can train with GPU using:")
        print("  python main.py train datasets/your_dataset.txt --use-gpu")
    else:
        print("⚠️  GPU acceleration is NOT available")
        print("\nTraining will use CPU (slower but functional)")
        print("\nTo enable GPU:")
        print("  1. Install PyTorch with ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
        print("  2. Or install DirectML: pip install torch-directml")
        print("  3. Ensure AMD GPU drivers are installed")
    print("=" * 70 + "\n")
    
    return 0 if (gpu_available or numpy_ok) else 1

if __name__ == "__main__":
    sys.exit(main())

