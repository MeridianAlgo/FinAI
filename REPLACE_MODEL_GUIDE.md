# How to Replace Model on Hugging Face

## Quick Method (Recommended)

1. **Set your Hugging Face token:**
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. **Run the upload script:**
   ```bash
   python force_upload_new_model.py
   ```

This will:
- Delete all old model files from Hugging Face
- Upload the new fresh model
- Update the README with v2.0 information

## Manual Method (Alternative)

1. Go to https://huggingface.co/MeridianAlgo/Fin.AI/tree/main
2. Delete old files: `model.pt`, `config.json`, `README.md`
3. Upload new files from `checkpoints/model/`

## After Upload

Once uploaded, GitHub Actions will automatically:
- Download this new model
- Continue training from it
- Upload updates after each training cycle

## Verify It Worked

1. Check https://huggingface.co/MeridianAlgo/Fin.AI
2. README should show "v2.0.0 - Fresh Start"
3. Model file should be ~115MB (30M parameters)
4. Config should show 6 layers, 384 embed_dim

## What Happens Next

- GitHub Actions will train this new model every 1h 10min
- Model starts from random weights (gibberish output initially)
- After 2-4 weeks of training, outputs should become coherent
- Each training run adds 800 steps of learning
