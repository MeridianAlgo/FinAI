---
language:
- en
license: mit
tags:
- transformer
- pytorch
- causal-lm
- finance
datasets:
- wikitext
- c4
pipeline_tag: text-generation
inference:
  parameters:
    temperature: 0.8
    max_length: 100
---

# Fin.AI

**WORK IN PROGRESS – EXPERIMENTAL RESEARCH PROJECT**

A continuously learning transformer language model that trains automatically every ~1.5 hours on diverse datasets using GitHub Actions.

> **Important Notice**  
> Fin.AI is an **experimental research prototype** and **work in progress**.  
> The model is under continuous training and may produce inaccurate, inappropriate, biased, or nonsensical outputs.  
> **Do NOT use for production applications, critical systems, or high-stakes decisions.**  
> Use at your own risk.

<div align="center">

[![Model on Hugging Face](https://img.shields.io/badge/🤗_Model-Fin.AI-yellow)](https://huggingface.co/MeridianAlgo/Fin.AI)
[![CI - Tests & Lint](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/ci.yml)
[![Training Workflow](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Overview

Fin.AI is an experimental GPT-style language model that trains **24/7** with a rotating curriculum of 24 different dataset families.

**Core characteristics:**

- Fully automated hourly training (GitHub Actions)
- 24 diverse dataset categories (news, math, code, dialogue, science, instructions...)
- Focus rotates every ~1.5 hours → targeted capability improvement
- Models automatically pushed to Hugging Face after each run
- Training metrics publicly visible on Weights & Biases
- Designed to run efficiently even on free GitHub runners

> This is **not** a production-ready model. Expect evolving (and sometimes unstable) behavior.

## Key Features

| Feature                  | Description                                                                |
|--------------------------|----------------------------------------------------------------------------|
| Automated Continuous Training | Trains every ~1.5 hours – completely hands-free                         |
| Rotating Curriculum      | 24 dataset families covering very different capabilities                  |
| Hugging Face Integration | Latest checkpoint pushed automatically after every training cycle        |
| Real-time Monitoring     | Full metrics, loss curves and samples on Weights & Biases                 |
| Flexible Scale           | Easily switch between ~15M and ~350M+ parameters                           |
| CPU-friendly             | Optimized to train efficiently on standard GitHub Actions runners         |

## Training Curriculum (24-cycle daily rotation)

| Category              | Example Datasets                     | Cycle Hours       | Primary Focus                     |
|-----------------------|--------------------------------------|-------------------|-----------------------------------|
| Encyclopedia          | WikiText                             | 0, 6              | Broad world knowledge             |
| Creative Writing      | TinyStories                          | 1, 18             | Storytelling & narrative          |
| News                  | CNN, AG News, CC News                | 2,15,17,20        | Current events & factual style    |
| Math & Reasoning      | GSM8K, CommonsenseQA                 | 3,9,19,23         | Problem solving & logic           |
| Open Web Text         | OpenWebText, C4                      | 4,11              | Diverse internet language         |
| Question Answering    | SQuAD                                | 5,22              | Reading comprehension             |
| Instruction Following | Alpaca, Dolly                        | 7,14,21           | Following user instructions       |
| Reviews & Sentiment   | IMDB, Amazon, Yelp                   | 8,10,16           | Opinion & sentiment analysis      |
| Scientific / Medical  | PubMed                               | 12                | Scientific & medical literature   |
| Conversations         | UltraChat                            | 13                | Natural dialogue                  |

## Quick Start

### Download Latest Model

```python
from huggingface_hub import hf_hub_download

hf_hub_download("MeridianAlgo/Fin.AI", "model.pt",   local_dir="./model")
hf_hub_download("MeridianAlgo/Fin.AI", "config.json", local_dir="./model")
```

### Basic Inference Example

```python
from fin_ai.model import FinAIModel
import torch

model = FinAIModel.from_pretrained("./model")
tokenizer = model.tokenizer

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_length=100, temperature=0.8)
print(tokenizer.decode(outputs[0]))
```

> **Warning**: Output quality is experimental and may contain factual errors, biases, or inappropriate content.

## Model Sizes (V3)

| Preset  | Parameters | Layers | Heads | Hidden Dim | Recommended Use Case             |
|---------|------------|--------|-------|------------|----------------------------------|
| tiny    | ~15M       | 6      | 4     | 256        | Very fast experiments            |
| small   | ~40M       | 8      | 8     | 512        | Default – good CPU performance   |
| medium  | ~120M      | 12     | 12    | 768        | Noticeably higher quality        |
| large   | ~350M      | 24     | 16    | 1024       | Best results (GPU recommended)   |

## Current Project Status

[![Status](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions)

- Latest checkpoint: [huggingface.co/MeridianAlgo/Fin.AI](https://huggingface.co/MeridianAlgo/Fin.AI)
- Training pipeline runs: [GitHub Actions](https://github.com/MeridianAlgo/FinAI/actions)
- Live metrics & samples: [Weights & Biases](https://wandb.ai/meridianalgo-meridianalgo/fin-ai)

---

<div align="center">

**Made with passion by the Fin.AI team**  
[⭐ Star on GitHub](https://github.com/MeridianAlgo/FinAI)  [🤗 View & download on Hugging Face](https://huggingface.co/MeridianAlgo/Fin.AI)

</div>
