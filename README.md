# Meridian.AI

A custom sparse Mixture-of-Experts language model continually trained on finance data. Runs automatically on GitHub Actions free runners — no GPU required.

**Model on Hugging Face:** [meridianal/FinAI](https://huggingface.co/meridianal/FinAI)

---

## Architecture

Meridian.AI is built from scratch — not a fine-tune of an existing model.

| Component | Detail |
|---|---|
| Type | Sparse Mixture-of-Experts (SMoE) |
| Layers | 14 transformer layers (alternating dense / MoE) |
| Attention | GQA — 12 query heads, 4 KV heads |
| Experts | 8 per MoE layer, top-2 active per token |
| Position encoding | RoPE (`theta=500k`) |
| Activation | SwiGLU |
| Normalisation | RMSNorm |
| Finance feature | Numeracy encoding (64-dim learned embedding for numeric tokens) |
| Continual learning | Elastic Weight Consolidation (EWC) |
| Total params | ~479M (tied embeddings) / ~283M unique |
| Tokenizer | Qwen2.5-0.5B (151k vocab) |
| Context | 2048 tokens |

---

## Repository Layout

```
FinAI/
├── meridian/
│   ├── model/
│   │   ├── configuration.py     # MeridianConfig
│   │   └── modeling.py          # MeridianForCausalLM
│   ├── data/
│   │   └── pipeline.py          # Finance dataset streaming
│   └── training/
│       └── trainer.py           # MeridianTrainer + EWC
├── scripts/
│   ├── seed_hf_repo.py          # Initialise HF repo with fresh weights
│   ├── evaluate_model.py        # Run eval prompts
│   └── ...
├── train.py                     # Main training entry point
├── requirements.txt
├── .github/
│   └── workflows/
│       └── train.yml            # Hourly CI training workflow
└── FinAI/
    └── README.md                # HuggingFace model card
```

---

## How It Trains

Training runs automatically every hour via GitHub Actions:

1. **Pull** — downloads the latest checkpoint from `meridianal/FinAI` on HuggingFace.
2. **Train** — streams finance datasets, runs up to 150 gradient steps with EWC regularisation.
3. **Upload** — pushes the updated checkpoint back to HuggingFace.
4. **Sync** — commits dataset state and formatting changes back to this repo.

The workflow is memory-safe: training halts and saves before hitting the ~14GB runner RAM ceiling. AdaFactor optimizer state is discarded before upload to keep checkpoint size manageable.

---

## Local Setup

```bash
git clone https://github.com/MeridianAlgo/FinAI.git
cd FinAI
pip install -r requirements.txt
```

### Quick smoke test

```bash
SMOKE_TEST=1 FAST_MODE=1 python train.py
```

### Full local training run

```bash
export HF_TOKEN=your_token_here
python train.py
```

**Key environment variables:**

| Variable | Default | Description |
|---|---|---|
| `MAX_STEPS` | 150 | Gradient steps per run |
| `BATCH_SIZE` | 2 | Per-step batch size |
| `GRAD_ACCUM` | 4 | Gradient accumulation steps |
| `BLOCK_SIZE` | 256 | Sequence length |
| `LEARNING_RATE` | 5e-5 | AdaFactor LR |
| `USE_EWC` | 1 | Enable EWC regularisation |
| `MAX_RAM_GB` | 13.0 | Hard RAM limit before emergency save |
| `SKIP_OPTIMIZER_SAVE` | 1 | Drop optimizer state before saving |
| `USE_LIGHT_DATASETS` | 0 | Use only small (<15MB) datasets |
| `FAST_MODE` | 0 | Minimal settings for quick debugging |

---

## Using the Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "meridianal/FinAI"

tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="checkpoint")
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder="checkpoint",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()

prompt = """### Instruction:
What is the difference between a bond's yield and its coupon rate?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
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

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## CI Setup (GitHub Actions)

Required repository secrets:

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace write token for `meridianal/FinAI` |
| `GH_PAT` | GitHub personal access token (for pushing commits) |
| `COMET_API_KEY` | (Optional) CometML experiment tracking |

The workflow runs hourly and can also be triggered manually from the Actions tab with optional `force_seed` (re-initialises the HF repo) and `max_steps` overrides.

---

## Disclaimer

Meridian.AI is an experimental research project. Outputs should not be used for real financial decisions. Not financial advice.
