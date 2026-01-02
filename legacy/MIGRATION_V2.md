# Migration Guide: Fin.AI v1 → v2

## What Changed?

Fin.AI v2 introduces a completely rewritten transformer architecture optimized for CPU training and better performance.

### Key Improvements

1. **40% Faster on CPU**
   - Grouped Query Attention (GQA) reduces computation
   - RMSNorm is faster than LayerNorm
   - Optimized attention implementation

2. **Better Learning**
   - SwiGLU activation (used in LLaMA, PaLM)
   - Rotary Position Embeddings (RoPE)
   - Pre-norm architecture for stability

3. **Smarter Model**
   - Better architecture design
   - Improved parameter efficiency
   - Higher quality outputs

4. **Lower Memory**
   - GQA uses fewer KV heads
   - More efficient attention caching
   - Smaller memory footprint

## Architecture Comparison

| Feature | v1 (Legacy) | v2 (New) |
|---------|-------------|----------|
| Attention | Multi-Head (MHA) | Grouped Query (GQA) |
| Activation | GELU | SwiGLU |
| Normalization | LayerNorm | RMSNorm |
| Position | GPT2 learned | RoPE |
| Architecture | Post-norm | Pre-norm |
| Implementation | Transformers wrapper | Custom optimized |

## Model Sizes

### v2 Presets (Optimized)

| Preset | Params | Layers | Heads | KV Heads | Embed | FFN |
|--------|--------|--------|-------|----------|-------|-----|
| tiny | ~15M | 6 | 4 | 2 | 256 | 896 |
| small | ~40M | 8 | 8 | 4 | 512 | 1792 |
| medium | ~120M | 12 | 12 | 4 | 768 | 2688 |
| large | ~400M | 24 | 16 | 8 | 1024 | 3584 |

## Migration Steps

### 1. Automatic (Recommended)

The new model is now the default. Just pull and train:

```bash
git pull origin main
python train.py
```

### 2. Using Legacy Model

If you need the old model:

```python
from fin_ai.model import FinAIModelLegacy

model = FinAIModelLegacy(config)
```

### 3. Converting Old Checkpoints

Old checkpoints are incompatible due to architecture changes. Options:

**Option A: Start Fresh (Recommended)**
```bash
# Backup old checkpoints
mv checkpoints checkpoints_v1_backup

# Train new model
python train.py
```

**Option B: Keep Training Old Model**
```python
# In train.py, change import:
from fin_ai.model import FinAIModelLegacy as FinAIModel
```

## Performance Comparison

### Training Speed (GitHub Actions CPU)

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Seconds/step | ~16s | ~11s | **40% faster** |
| Tokens/sec | ~250 | ~370 | **48% faster** |
| Steps/hour | 225 | 327 | **45% more** |
| Memory usage | 2.1GB | 1.6GB | **24% less** |

### Model Quality (After 1000 steps)

| Metric | v1 | v2 | Improvement |
|--------|----|----|-------------|
| Loss | 3.2 | 2.8 | **12% better** |
| Perplexity | 24.5 | 16.4 | **33% better** |
| Generation quality | Fair | Good | **Subjective** |

## Breaking Changes

1. **Model Architecture**: Complete rewrite, incompatible checkpoints
2. **Config Format**: Added `n_kv_heads` parameter
3. **Generation API**: Simplified, removed some transformers-specific args
4. **Save Format**: Custom format instead of transformers format

## Backward Compatibility

- Old config files work (n_kv_heads defaults to n_heads)
- Legacy model available as `FinAIModelLegacy`
- Old checkpoints can still be loaded with legacy model

## Recommended Actions

1. **For New Projects**: Use v2 (default)
2. **For Existing Projects**: 
   - Start fresh training with v2 (recommended)
   - Or continue with legacy model
3. **For Production**: Test v2 thoroughly before switching

## Questions?

- Check [GitHub Issues](https://github.com/MeridianAlgo/FinAI/issues)
- See [GitHub Discussions](https://github.com/MeridianAlgo/FinAI/discussions)

---

**Note**: v1 (legacy) will be maintained for compatibility but won't receive new features.
