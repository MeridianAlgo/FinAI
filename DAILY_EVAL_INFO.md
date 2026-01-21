# Daily Model Evaluation

## Overview

The daily evaluation workflow tests the model with the same prompt every day to track how its responses evolve as it trains continuously (hourly).

## How It Works

1. **Schedule**: Runs daily at 6 AM UTC (after several hourly training runs)
2. **Test Prompt**: "The future of artificial intelligence is"
3. **Process**:
   - Downloads latest model from Hugging Face
   - Generates response with consistent parameters
   - Saves to `daily_eval_history.json` (keeps last 30 days)
   - Updates README with last 7 days of responses
   - Commits changes back to repo

## Viewing Results

- **README**: Check the "Daily Model Evolution" section
- **History File**: `daily_eval_history.json` contains full 30-day history
- **Badge**: Daily Evaluation badge shows workflow status

## Manual Testing

You can run the evaluation manually:

```bash
# Test local model
python scripts/daily_eval.py --model-path checkpoints/model

# Test model from Hugging Face
python scripts/daily_eval.py --hf-repo MeridianAlgo/Fin.AI

# Custom prompt
python scripts/daily_eval.py --prompt "Once upon a time" --max-new-tokens 150
```

## Workflow Trigger

- **Automatic**: Daily at 6 AM UTC
- **Manual**: Go to Actions → Daily Model Evaluation → Run workflow

## What to Expect

As the model trains hourly on diverse datasets, you should see:
- Improved coherence over time
- Better grammar and structure
- More contextually relevant responses
- Evolution of "personality" based on training data

## Notes

- Workflow uses `[skip ci]` to avoid triggering other workflows
- History is capped at 30 days to keep file size manageable
- README shows only last 7 days for readability
- Same generation parameters used each time for consistency
