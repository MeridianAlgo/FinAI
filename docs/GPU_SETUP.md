# GPU Setup Guide for AMD Radeon RX 7600XT

This guide will help you set up GPU acceleration for training FinAI on your AMD Radeon RX 7600XT GPU.

## Quick Start

For **Windows** with AMD RX 7600XT, you have several options:

### Option 1: PyTorch with ROCm (Recommended for Windows)

AMD now provides a preview version of PyTorch with ROCm support for Windows:

```bash
# Install PyTorch with ROCm support (Windows preview)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Requirements:**
- Windows 10/11 (64-bit)
- AMD Radeon RX 7600XT (or other supported AMD GPUs)
- AMD GPU drivers installed

### Option 2: DirectML (Alternative for Windows)

DirectML is Microsoft's DirectX-based machine learning framework that works with AMD GPUs on Windows:

```bash
# Install PyTorch with DirectML support
pip install torch-directml
```

This is simpler to set up and often works out-of-the-box on Windows.

### Option 3: CPU-only (Fallback)

If GPU setup fails, PyTorch will automatically fall back to CPU:

```bash
# Standard PyTorch (CPU)
pip install torch torchvision torchaudio
```

## Verification

After installing PyTorch, verify GPU detection:

```python
import torch

# Check CUDA/ROCm
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
else:
    print("No GPU detected")

# Check DirectML (if installed)
try:
    import torch_directml
    if torch_directml.is_available():
        print(f"DirectML device available: {torch_directml.device()}")
except ImportError:
    pass
```

## Using GPU for Training

When training, the system will automatically detect and use your GPU:

```bash
# Train with GPU acceleration (automatic detection)
python main.py train datasets/your_dataset.txt --use-gpu

# Or force CPU only
python main.py train datasets/your_dataset.txt --no-gpu
```

## Performance Tips

1. **Batch Size**: GPU training benefits from larger batch sizes. Try increasing `BATCH_SIZE` in `src/config.py` (e.g., 2048 or 4096) if you have enough VRAM.

2. **Mixed Precision**: The PyTorch model can use mixed precision training for faster training. This is automatically enabled when using GPU.

3. **Monitor GPU Usage**: Use tools like:
   - Windows Task Manager → Performance → GPU
   - `nvidia-smi` (if using NVIDIA GPU)
   - AMD Software: Adrenalin Edition

## Troubleshooting

### GPU Not Detected

1. **Check Drivers**: Ensure you have the latest AMD GPU drivers installed
   - Download from: https://www.amd.com/en/support

2. **Verify Installation**:
   ```python
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   ```

3. **Try DirectML**: If ROCm doesn't work, try DirectML as an alternative

### Out of Memory Errors

If you encounter GPU memory errors:

1. Reduce `BATCH_SIZE` in `src/config.py`
2. Reduce `MAX_SEQUENCE_LENGTH` in `src/config.py`
3. Use streaming training: `--stream` flag

### Slow Performance

If training is still slow:

1. Verify GPU is being used (check Task Manager)
2. Increase batch size (if memory allows)
3. Check that data loading isn't the bottleneck
4. Ensure you're using the PyTorch model, not the scikit-learn model

## System Requirements

- **OS**: Windows 10/11 (64-bit) or Linux
- **GPU**: AMD Radeon RX 7600XT or compatible
- **RAM**: 8GB minimum, 16GB recommended
- **VRAM**: 8GB+ recommended for larger models
- **Python**: 3.9+

## Expected Speedup

With GPU acceleration, you can expect:
- **5-20x faster** training compared to CPU
- **Varies** based on:
  - Model size (hidden_dim, vocab_size)
  - Batch size
  - Dataset size
  - GPU specifications

## Next Steps

1. Install PyTorch with GPU support (see options above)
2. Verify GPU detection (run verification script)
3. Train a small dataset first to test
4. Monitor GPU usage during training
5. Adjust batch size and other hyperparameters for optimal performance

## Additional Resources

- PyTorch ROCm: https://pytorch.org/get-started/locally/
- DirectML: https://github.com/microsoft/DirectML
- AMD ROCm: https://rocm.docs.amd.com/

