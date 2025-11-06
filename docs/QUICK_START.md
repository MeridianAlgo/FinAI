# ⚡ FinAI Quick Start Guide

## 🚀 Training Options

### **Option 1: Sequential Training (Recommended)**
Train datasets one at a time - see progress after each dataset!

```bash
python train_sequential.py
```

**Perfect for:**
- ✅ Multiple datasets
- ✅ Want to see progress incrementally
- ✅ Can stop/resume anytime
- ✅ Git commits after each dataset

---

### **Option 2: Combined Training**
Train all datasets together in one session.

```bash
python train_all.py
```

**Perfect for:**
- ✅ Overnight training
- ✅ All datasets at once
- ✅ Shows ETA during training

---

## 📊 What You'll See During Training

```
================================================================================
FinAI Training - Single Model Continuous Learning
================================================================================
Loading dataset from combined_training_data.txt...
✓ Loaded existing tokenizer
✓ Tokenized 1,234,567 tokens

✓ Loading existing model to CONTINUE training...
  Previous training: 10,000 steps completed

================================================================================
Training Configuration:
  Device: cuda
  Steps: 10000 | Batch size: 32 | Block size: 1024
  Learning rate: 0.0006 | Warmup steps: 100
  Gradient accumulation: 8 (effective batch: 256)
  Weight decay: 0.1 | Max grad norm: 1.0
================================================================================

Step 500/10000 | loss 2.3456 | lr 6.00e-04 | elapsed 0:05:23 | ETA 0:48:12
Step 1000/10000 | loss 2.1234 | lr 5.95e-04 | elapsed 0:10:45 | ETA 0:42:30
Step 1500/10000 | loss 1.9876 | lr 5.88e-04 | elapsed 0:16:08 | ETA 0:36:52
...
```

---

## 📁 Setup Your Datasets

Edit `datasets.csv`:

```csv
name,config,split,date_trained,model_path,status
virattt/financial-qa-10K,,train,,,
gbharti/finance-alpaca,,train,,,
FinanceInc/auditor_sentiment,,train,,,
```

Then run:
```bash
python train_sequential.py
```

---

## 🔄 Resume Training

**Training automatically resumes!**

If you stop training (Ctrl+C) and run again:
```bash
python train_sequential.py
```

It will:
1. Load the existing model
2. Check `trained_datasets.csv` for completed datasets
3. Continue with the next pending dataset

---

## 💬 Chat with Your Model

```bash
python main.py chat
```

Example:
```
You: What is portfolio diversification?
FinAI: Portfolio diversification is an investment strategy...
```

---

## 📈 Track Progress

### **datasets.csv** (Pending)
```csv
name,config,split,date_trained,model_path,status
virattt/financial-qa-10K,,train,,,
```

### **trained_datasets.csv** (Completed)
```csv
name,config,split,date_trained,model_path,status
gbharti/finance-alpaca,,train,2024-11-05 18:45:23,models/finai_gpt.pt,completed
```

### **Git History**
```bash
git log --oneline
```
```
abc1234 Model Dataset #3 Trained
def5678 Model Dataset #2 Trained
ghi9012 Model Dataset #1 Trained
```

---

## ⚙️ Configuration

Edit `src/config.py` to adjust:

```python
# Quick adjustments for 8GB GPU:
N_LAYER = 12               # Reduce from 24
N_EMBD = 768               # Reduce from 1024
BATCH_SIZE = 16            # Reduce from 32
GRADIENT_ACCUM_STEPS = 16  # Increase to compensate
```

---

## 🆘 Common Issues

### **Out of Memory**
```python
# In src/config.py:
BATCH_SIZE = 16  # Reduce batch size
N_LAYER = 12     # Reduce model size
```

### **Training Too Slow**
- Already using bfloat16 mixed precision ✓
- Already using gradient checkpointing ✓
- Already using Flash Attention ✓
- Consider reducing `TRAIN_STEPS` for faster iterations

### **Model Not Generating Well**
- Train longer (20k-50k steps)
- Add more diverse datasets
- Lower temperature (0.5-0.7)

---

## 📚 Full Documentation

See `README.md` for complete documentation.

---

**Start training now:**
```bash
python train_sequential.py
```

**Happy training! 🚀**
