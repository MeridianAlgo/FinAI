# 🎓 FinAI Training Guide

## 🎯 Two Training Modes

### **Mode 1: Sequential Training (Recommended)**
**File:** `train_sequential.py`

Trains datasets **ONE AT A TIME** - perfect for seeing progress!

```bash
python train_sequential.py
```

**How it works:**
1. Loads first pending dataset from `datasets.csv`
2. Trains on that dataset only (10,000 steps by default)
3. Saves model checkpoint
4. Commits to git: `"Model Dataset #1 Trained"`
5. Moves dataset to `trained_datasets.csv`
6. **Repeats for next dataset**

**Advantages:**
- ✅ See progress after each dataset
- ✅ Can stop/resume anytime
- ✅ Git history shows each dataset
- ✅ Faster feedback loop
- ✅ Less risk if training fails

**Best for:**
- Multiple datasets
- Want incremental progress
- Limited training time per session

---

### **Mode 2: Combined Training**
**File:** `train_all.py`

Trains **ALL DATASETS TOGETHER** in one session.

```bash
python train_all.py
```

**How it works:**
1. Loads ALL pending datasets from `datasets.csv`
2. Combines them into one large text file
3. Trains on combined data (10,000 steps)
4. Saves model checkpoint
5. Commits to git for each dataset

**Advantages:**
- ✅ One long training session
- ✅ All datasets processed together
- ✅ Shows ETA during training

**Best for:**
- Overnight training
- Want to process all datasets at once
- Have stable power/internet

---

## 📊 Training Progress Display

Both modes show detailed progress:

```
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
```

**Understanding the output:**
- **Step**: Current/Total steps
- **loss**: Training loss (aim for <2.0 for good results)
- **lr**: Learning rate (decreases over time with cosine schedule)
- **elapsed**: Time spent so far
- **ETA**: Estimated time remaining (updates dynamically)

---

## 🔄 Checkpoint Resumption

**Training automatically resumes from where you left off!**

### **How it works:**

The model saves:
```python
{
    'model_state_dict': ...,      # Model weights
    'training_state': {
        'total_steps_completed': 10000,  # Cumulative steps
        'datasets_trained': 2,            # Number of datasets
    }
}
```

### **Example workflow:**

```bash
# Session 1: Train on dataset #1
python train_sequential.py
# → Trains 10,000 steps
# → Total steps: 10,000
# → Commits: "Model Dataset #1 Trained"

# Session 2: Train on dataset #2
python train_sequential.py
# → Loads existing model (10,000 steps)
# → Trains another 10,000 steps
# → Total steps: 20,000
# → Commits: "Model Dataset #2 Trained"

# Session 3: Train on dataset #3
python train_sequential.py
# → Loads existing model (20,000 steps)
# → Trains another 10,000 steps
# → Total steps: 30,000
# → Commits: "Model Dataset #3 Trained"
```

**The model gets better with each dataset!**

---

## 📁 Dataset Tracking

### **datasets.csv** (Pending datasets)
```csv
name,config,split,date_trained,model_path,status
virattt/financial-qa-10K,,train,,,
gbharti/finance-alpaca,,train,,,
FinanceInc/auditor_sentiment,,train,,,
```

### **trained_datasets.csv** (Completed datasets)
```csv
name,config,split,date_trained,model_path,status
virattt/financial-qa-10K,,train,2024-11-05 17:30:42,models/finai_gpt.pt,completed
gbharti/finance-alpaca,,train,2024-11-05 18:45:23,models/finai_gpt.pt,completed
```

**Automatic tracking:**
- When a dataset completes, it moves from `datasets.csv` → `trained_datasets.csv`
- Timestamp is added automatically
- Model path is recorded
- Status is set to "completed"

---

## 🔒 Git Automation

After each dataset trains:

```bash
git add .
git commit -m "Model Dataset #1 Trained"
git push origin main
```

**Git history:**
```bash
$ git log --oneline
abc1234 Model Dataset #3 Trained
def5678 Model Dataset #2 Trained  
ghi9012 Model Dataset #1 Trained
```

**Setup (first time only):**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git remote add origin https://github.com/your-username/FinAI.git
```

---

## ⏱️ Training Time Estimates

**Per dataset (10,000 steps):**

| GPU | Batch Size | Time per Dataset |
|-----|------------|------------------|
| AMD RX 7600 XT (16GB) | 32 | ~2-3 hours |
| NVIDIA RTX 3090 (24GB) | 32 | ~1.5-2 hours |
| NVIDIA RTX 4090 (24GB) | 64 | ~1-1.5 hours |
| CPU only | 8 | ~12-24 hours |

**Factors affecting speed:**
- Dataset size (more tokens = slower)
- GPU memory (larger batch = faster)
- Gradient accumulation steps
- Model size (24 layers vs 12 layers)

---

## ⚙️ Adjusting Training Speed

### **Faster training (less accuracy):**
```python
# In src/config.py:
TRAIN_STEPS = 5000         # Half the steps
BATCH_SIZE = 64            # Larger batches (if GPU allows)
GRADIENT_ACCUM_STEPS = 4   # Less accumulation
```

### **Better quality (slower):**
```python
# In src/config.py:
TRAIN_STEPS = 20000        # More steps
BATCH_SIZE = 16            # Smaller batches (more stable)
GRADIENT_ACCUM_STEPS = 16  # More accumulation
```

### **For 8GB GPU:**
```python
# In src/config.py:
N_LAYER = 12               # Smaller model
N_EMBD = 768               # Smaller embeddings
BATCH_SIZE = 16            # Smaller batch
GRADIENT_ACCUM_STEPS = 16  # Compensate with accumulation
```

---

## 🎯 Recommended Workflow

### **For multiple datasets:**

1. **Add all datasets to `datasets.csv`**
   ```csv
   virattt/financial-qa-10K,,train,,,
   gbharti/finance-alpaca,,train,,,
   FinanceInc/auditor_sentiment,,train,,,
   ```

2. **Run sequential training**
   ```bash
   python train_sequential.py
   ```

3. **Let it run!**
   - Trains dataset #1 → commits → continues
   - Trains dataset #2 → commits → continues
   - Trains dataset #3 → commits → done

4. **Check progress**
   ```bash
   cat trained_datasets.csv
   git log --oneline
   ```

5. **Chat with your model**
   ```bash
   python main.py chat
   ```

---

## 🆘 Troubleshooting

### **Training interrupted?**
Just run again - it will resume!
```bash
python train_sequential.py
```

### **Out of memory?**
Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 16  # or 8
```

### **Training too slow?**
- Check GPU is being used: `python scripts/verify_gpu.py`
- Reduce model size (see above)
- Use fewer training steps

### **Loss not decreasing?**
- Train longer (20k steps)
- Check learning rate (6e-4 is good)
- Ensure data quality is good

### **Git push failing?**
```bash
# Check remote
git remote -v

# Add remote if missing
git remote add origin https://github.com/your-username/FinAI.git

# Push manually
git push -u origin main
```

---

## 📚 Next Steps

1. **Train your first dataset**
   ```bash
   python train_sequential.py
   ```

2. **Monitor progress**
   - Watch the ETA
   - Check loss decreasing
   - Wait for git commit

3. **Test your model**
   ```bash
   python main.py chat
   ```

4. **Add more datasets**
   - Edit `datasets.csv`
   - Run training again

5. **Share your model**
   - Push to GitHub
   - Model is in `models/finai_gpt.pt`

---

**Happy training! 🚀**

For more details, see `README.md` or `QUICK_START.md`
