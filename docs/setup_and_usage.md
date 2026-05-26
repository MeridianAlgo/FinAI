# Setup and Usage

## Requirements

- Python 3.10 or higher
- ~2 GB disk (for model weights from HuggingFace)
- ~4 GB RAM minimum for inference; 8+ GB recommended for training

---

## Installation

```bash
git clone https://github.com/MeridianAlgo/FinAI.git
cd FinAI
pip install -r requirements.txt
```

### Verify Installation

Run the smoke test — no downloads required, runs a tiny in-memory model:

```bash
SMOKE_TEST=1 FAST_MODE=1 python train.py
```

Expected: `[OK] Smoke test passed!`

---

## Inference

### Option 1: From HuggingFace Hub (Recommended)

The latest trained checkpoint is always at `meridianal/FinAI` on HuggingFace.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "meridianal/FinAI"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="checkpoint")
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder="checkpoint",
    # trust_remote_code=True is NOT needed — this is standard Qwen2, not a custom arch
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()
```

### Option 2: From Local Checkpoint

After running `python train.py` locally, load from `./checkpoint`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint")
model.eval()
```

### Generating Text

Use the `### Instruction: / ### Response:` format that matches the training data:

```python
prompt = """### Instruction:
What does the price-to-earnings ratio tell an investor?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### Recommended Generation Parameters

| Parameter | Value | Purpose |
|:---|:---|:---|
| `temperature` | 0.7–0.9 | Controls randomness. Lower = more deterministic. |
| `top_p` | 0.90–0.95 | Nucleus sampling cutoff. |
| `repetition_penalty` | 1.2–1.4 | Discourages repeated phrases. |
| `no_repeat_ngram_size` | 3 | Hard block on 3-gram repeats. |
| `max_new_tokens` | 150–300 | Token budget for the response. |

---

## Local Training

### Full Training Run

```bash
export HF_TOKEN=your_huggingface_token
python train.py
```

This will:
1. Pull the latest checkpoint from `meridianal/FinAI` on HuggingFace
2. Load Qwen2.5-0.5B (or resume from checkpoint if architecture matches)
3. Stream financial datasets and train for 150 steps (default, v6.0.0)
4. Save the checkpoint locally and upload back to HuggingFace

### Training Without HuggingFace

Skip the checkpoint sync and train offline:

```bash
# No HF_TOKEN needed — starts fresh from Qwen2.5-0.5B
MAX_STEPS=50 python train.py
```

Model weights are still downloaded from HuggingFace on first run (Qwen2.5-0.5B base). Use `--local-files-only` via `TRANSFORMERS_OFFLINE=1` if working fully offline with a pre-cached model.

### Fast Debug Mode

Minimal settings for rapid local testing (no dataset streaming, tiny sequences):

```bash
FAST_MODE=1 python train.py
```

This sets: `USE_LIGHT_DATASETS=1`, `MAX_STEPS=5`, `BATCH_SIZE=1`, `GRAD_ACCUM=1`, `BLOCK_SIZE=32`, `USE_EWC=0`.

### Custom Configuration

```bash
MAX_STEPS=300 \
BATCH_SIZE=1 \
GRAD_ACCUM=4 \
LEARNING_RATE=3e-5 \
BLOCK_SIZE=512 \
USE_EWC=1 \
python train.py
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just model architecture tests
pytest tests/test_model.py -v

# Just trainer tests
pytest tests/test_training.py -v
```

Expected: all tests pass in ~30–60 seconds on CPU.

---

## Code Examples

Annotated scripts are in `examples/`:

| Script | What It Demonstrates |
|:---|:---|
| `01_inference.py` | Full HuggingFace inference pipeline from Hub |
| `02_dataset_pipeline.py` | Dataset streaming, curriculum weights, data preview |
| `03_model_config.py` | Direct instantiation of `MeridianConfig` + `MeridianForCausalLM` |

Run any example from the repo root:

```bash
python examples/01_inference.py
python examples/02_dataset_pipeline.py
python examples/03_model_config.py
```

---

## Environment Setup for Development

For development with linting and formatting:

```bash
pip install -r requirements.txt
pip install ruff black pytest pytest-cov

# Format
black .

# Lint
ruff check . --fix

# Type check
mypy meridian/ --ignore-missing-imports

# Tests with coverage
pytest tests/ --cov=meridian --cov-report=term-missing
```

---

## Disclaimer

Meridian.AI is experimental research software. Do not use model outputs for real financial decisions. This is not financial advice.
