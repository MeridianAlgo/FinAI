---
language:
- en
license: mit
library_name: transformers
tags:
- finai
- bitnet
- ternary
- liquid-neural-networks
- sub-quadratic
- finance
datasets:
- HuggingFaceFW/fineweb-edu
metrics:
- perplexity
pipeline_tag: text-generation
---

# FinAI-Next (331M) - Liquid-BitNet Model Card

FinAI-Next is a specialized Small Language Model (SLM) designed for high-efficiency financial reasoning. It utilizes a Liquid-BitNet architecture, which combines 1.58-bit ternary quantization with constant-time stateful recurrence based on Liquid Neural Network principles.

## Model Summary

- **Developer:** MeridianAlgo
- **Model Type:** Causal Language Model
- **Language:** English
- **License:** MIT
- **Parameter Count:** 331,296,816
- **Architecture Details:**
  - **Precision:** 1.58-bit (Ternary weights {-1, 0, 1})
  - **Sequence Modeling:** Liquid Dynamical Systems (Linear Complexity)
  - **Hidden Size:** 1536
  - **Depth:** 24 Layers
  - **Vocabulary Size:** 151,665

## Technical Specifications

### Architecture
The model relies on BitNet b1.58 technology, which replaces traditional floating-point multiplications with integer additions and subtractions. For sequence modeling, it replaces the standard Attention mechanism with Liquid Blocks, which maintain $O(n)$ complexity with respect to sequence length. This design facilitates processing long documents (32k+ context) on memory-constrained hardware.

### Training Data
The model is trained on the Fineweb-Edu dataset, a highly curated collection of educational content. The training process focuses on developing robust reasoning capabilities applicable to financial and technical domains.

### Training Hardware and Pipeline
- **Hardware:** Distributed between GitHub Actions (CPU-based) and local consumer-grade CPU environments.
- **Pipeline:** Continuous hourly training cycles with automated checkpointing and state persistence.
- **Precision:** Mixed-precision training with FP32 master weights and ternary quantized forward weights.

## Intended Use Cases

- **Financial Document Analysis:** Summarization and information extraction from audits, reports, and financial filings.
- **Edge Computing:** Deployment on systems lacking dedicated GPU acceleration, such as standard enterprise laptops and mobile devices.
- **Real-time Financial Reasoning:** Low-latency response generation for financial queries.

## Limitations and Ethical Considerations

- **Knowledge Domain:** Primary optimization for financial and educational reasoning. Performance on general creative tasks may be reduced compared to dense Transformer models.
- **Quantization Effects:** While ternary weights offer extreme efficiency, they may require longer training durations to achieve comparable perplexity to full-precision counterparts.
- **Bias:** Inherits biases present in the Fineweb-Edu training corpus.

## Implementation Details

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# The model requires the fin_ai repository code to be present in the python path
tokenizer = AutoTokenizer.from_pretrained("MeridianAlgo/FinAI-Lite")
model = AutoModelForCausalLM.from_pretrained("MeridianAlgo/FinAI-Lite", trust_remote_code=True)

prompt = "Analysis of current market liquidity:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Citation Information

```bibtex
@software{finai_next_2026,
  author = {Ishaan and MeridianAlgo Team},
  title = {FinAI-Next: Frontier Liquid-BitNet Architecture},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/MeridianAlgo/FinAI}}
}
```
