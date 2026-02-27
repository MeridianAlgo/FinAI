---
license: mit
base_model: hpcai-tech/openmoe-base
tags:
  - finance
  - mixture-of-experts
  - openmoe
  - umt5
language:
  - en
---

# Meridian.AI

Meridian.AI is an experimental finance-focused sparse Mixture-of-Experts (MoE) causal language model trained via continual updates.

This repository is designed to run on commodity CPU hardware (including GitHub Actions runners) and continuously improve the model over time.

## Intended use

Use this model for:

- Financial Q&A style prompting
- Basic quantitative finance explanations
- Summarization/classification-style finance text tasks

This model is intended for research and prototyping. It is not intended to provide financial advice.

## Base model + tokenizer

- **Base model weights**: `hpcai-tech/openmoe-base`
- **Tokenizer**: `google/umt5-small` (256k vocab, SentencePiece/umT5)

This repo includes a working umT5 tokenizer at the root so `AutoTokenizer.from_pretrained("MeridianAlgo/MeridianAI")` works.

## Architecture overview

Meridian.AI is a sparse Mixture-of-Experts (MoE) transformer:

- **Sparse routing**: only a small subset of expert parameters are activated per token.
- **Grouped Query Attention (GQA)**: reduces attention compute cost.
- **RoPE**: rotary positional embeddings.
- **Numeracy features**: additional components intended to improve numeric reasoning on finance tasks.

Even when a model is trained for finance, general text quality depends heavily on the base model and the stability of continual training.

## How to use (Transformers)

The model weights are stored under the `checkpoint/` subfolder.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "MeridianAlgo/MeridianAI"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    subfolder="checkpoint",
    trust_remote_code=True,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    ignore_mismatched_sizes=True,
)
model.eval()

prompt = """### Instruction:
Explain what a P/E ratio is and how it is used.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.85,
    top_p=0.92,
    repetition_penalty=1.25,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Generation tips (reduce repetition)

Because continual training can introduce repetition loops, start with:

- `repetition_penalty=1.2` to `1.4`
- `no_repeat_ngram_size=3`
- `temperature=0.7` to `0.95`
- `top_p=0.85` to `0.95`

If you see repeated tokens/phrases, increase `repetition_penalty` and decrease `temperature`.

## Training data

Training uses streaming mixes of finance datasets (FinanceMTEB family) plus optional larger corpora depending on environment configuration.

### Lightweight mode

To keep each dataset small (e.g. under ~15MB) and avoid large downloads, the code supports a light-datasets-only mode:

- `USE_LIGHT_DATASETS=1`

This uses a curated set of small FinanceMTEB datasets (sentiment, ESG, FOMC, fraud, complaints, small QA).

### Formatting

Instruction-style datasets are formatted as:

```
### Instruction:
<instruction>

### Response:
<response><eos>
```

Classification datasets are converted into instruction/response examples with a label-only response.

EOS tokens are appended to help the model learn when to stop generation.

## Training on GitHub Actions (memory-safe)

GitHub-hosted runners have limited RAM. The trainer supports the following environment variables:

- `SKIP_OPTIMIZER_SAVE=1` (recommended): avoids saving the 2GB+ optimizer state.
- `MAX_RAM_PCT=90` or `MAX_RAM_GB=14`: will save a weights-only checkpoint and stop before an OOM.

If you enable gradient checkpointing, `use_cache` is automatically disabled.

### Recommended Actions runner settings

If you are using a 2-core / 16GB runner, a conservative starting point:

- `BATCH_SIZE=1`
- `GRAD_ACCUM=4`
- `BLOCK_SIZE=128` to `256`
- `LEARNING_RATE=1e-5`
- `MAX_STEPS=20` to `50`

If you still see OOMs, reduce `BLOCK_SIZE` first.

### Checkpoint layout on the Hub

This Hugging Face repo contains:

- Root tokenizer files (umt5)
- A `checkpoint/` folder with the latest model weights + config

This means:

- Tokenizer: `AutoTokenizer.from_pretrained("MeridianAlgo/MeridianAI")`
- Model: `AutoModelForCausalLM.from_pretrained("MeridianAlgo/MeridianAI", subfolder="checkpoint", ...)`

## Evaluation

If you want to track whether training is improving (and not collapsing), periodically run a fixed set of prompts and record:

- Repetition rate (e.g. fraction of repeated 3-grams)
- Numeric accuracy on a small set of math/finance questions
- Qualitative Q&A quality

The repo includes scripts to run basic prompt tests.

## Limitations

- This is a continually trained experimental model and may exhibit repetition.
- Not financial advice.
- Outputs may be incorrect or outdated.

## Roadmap

Potential next improvements:

- Add stronger evaluation gates to prevent uploading collapsed checkpoints
- Add curated finance instruction sets (filings, earnings calls, QA)
- Improve chat formatting / system prompts
- Add safe serialization and sharding options for faster uploads

## Source code

Training pipeline and scripts live in the GitHub repo: https://github.com/MeridianAlgo/FinAI
