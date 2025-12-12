<div align="center">

# 🤖 FinAI

### *A Self-Training Financial Language Model*
### *Powered by EfficientFinAI Architecture*

<br>

[![Training](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml/badge.svg)](https://github.com/MeridianAlgo/FinAI/actions/workflows/train.yml)
[![Release](https://img.shields.io/github/v/release/MeridianAlgo/FinAI?include_prereleases&label=latest)](https://github.com/MeridianAlgo/FinAI/releases)
[![License](https://img.shields.io/github/license/MeridianAlgo/FinAI)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Model](https://img.shields.io/badge/model-~12M_params-blueviolet)](https://github.com/MeridianAlgo/FinAI)

<br>

**⚠️ UNDER ACTIVE DEVELOPMENT ⚠️**

*This model is continuously training and improving. Expect frequent updates.*

<br>

[**Getting Started**](#-quick-start) • [**Releases**](https://github.com/MeridianAlgo/FinAI/releases) • [**Training Status**](#-live-training-status)

---

<br>

## 📊 Live Training Status

<br>

| Metric | Status |
|:------:|:------:|
| 🔄 **Training** | ![Status](https://img.shields.io/badge/status-active-brightgreen) |
| 📈 **Total Steps** | ![Steps](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/MeridianAlgo/FinAI/master/training_state.json&query=$.total_steps&label=&color=blue) |
| 🔁 **Cycles** | ![Cycles](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/MeridianAlgo/FinAI/master/training_state.json&query=$.cycle_count&label=&color=purple) |
| 🏷️ **Version** | ![Version](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/MeridianAlgo/FinAI/master/training_state.json&query=$.version&label=v&color=orange) |
| 📦 **Releases** | ![Releases](https://img.shields.io/badge/dynamic/json?url=https://raw.githubusercontent.com/MeridianAlgo/FinAI/master/training_state.json&query=$.releases_created&label=&color=red) |
| 📚 **Datasets** | ![Datasets](https://img.shields.io/badge/14-datasets-green) |

<br>

### ⏰ Training Schedule (UTC)

| 04:00 | 06:00 | 08:00 | 10:00 | 12:00 | 14:00 | 16:00 | 18:00 | 20:00 | 22:00 | 23:00-04:00 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|
| 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 🔴 Paused |

<br>

---

<br>

## 🚀 Quick Start

```bash
git clone https://github.com/MeridianAlgo/FinAI.git
cd FinAI
pip install -r requirements.txt
python main.py
```

<br>

---

<br>

## 🏗️ Model Architecture

**EfficientFinAI** - Optimized for CPU training on GitHub Actions

| Component | Technology | Benefit |
|:---------:|:----------:|:-------:|
| **Positional Encoding** | RoPE (Rotary) | Better long-context understanding |
| **Activation** | SwiGLU | 10% more efficient than GELU |
| **Normalization** | RMSNorm | Faster than LayerNorm |
| **Attention** | Flash Attention 2 | Memory-efficient |
| **Weight Tying** | Embeddings ↔ Output | Reduced parameters |

**Model Specs:**
- 📊 ~12M parameters
- 🧠 6 layers, 6 heads
- 📏 384 embedding dimensions
- 📖 512 token context window

<br>

---

<br>

## 📚 Training Datasets

| Dataset | Type | Status |
|:-------:|:----:|:------:|
| `finance-alpaca` | Q&A | ✅ |
| `financebench` | Financial Q&A | ✅ |
| `fingpt-forecaster` | Forecasting | ✅ |
| `reddit_finance_posts` | Reddit | ✅ |
| `twitter-financial-news` | Sentiment | ✅ |
| `FinanceQA` | Q&A | ✅ |
| `Quant-Trading-Instruct` | Trading | ✅ |
| `finer-ord` | NER | ✅ |
| `trade-the-event` | Events | ✅ |
| `auditor_sentiment` | Sentiment | ✅ |
| `finance_dataset` | General | ✅ |
| `chatgpt-prompts` | Prompts | ✅ |
| `agent-finance-reasoning` | Reasoning | ✅ |
| `english_quotes` | Quotes | ✅ |

**14 datasets • Auto-cycles after completion • Tracks success/failure**

<br>

---

<br>

## ⚙️ How It Works

```
┌────────────────────────────────────────────────────────────────┐
│                      TRAINING CYCLE                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Dataset 1 ──► Dataset 2 ──► ... ──► Dataset 14 ──► Cycle!    │
│  (5k steps)    (5k steps)           (5k steps)                │
│                                                                │
│  ✅ Success → Track progress → Continue next run              │
│  ❌ Failure → Log error → Skip to next dataset                │
│                                                                │
│  Every 3 cycles ──► New Release (v1.0.x) + Test Results       │
│                                                                │
│  After all datasets complete ──► Recycle from beginning       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

<br>

---

<br>

## 📦 Releases

Every **3 training cycles**, a new release is created:

- 📥 Model weights (`finai_gpt.pt`)
- 📊 Training statistics
- 🧪 10 sample Q&A test results
- 📈 Per-dataset progress

<br>

[**➡️ View All Releases**](https://github.com/MeridianAlgo/FinAI/releases)

<br>

---

<br>

## 📁 Structure

```
FinAI/
├── main.py               # Model & chat
├── train_cycle.py        # Training loop
├── training_state.json   # Global state
├── datasets.csv          # Dataset config
├── trained_datasets.csv  # Per-dataset progress
├── models/               # Saved weights
└── src/                  # Source code
```

<br>

---

<br>

## 🛠️ Development Status

| Component | Status |
|:---------:|:------:|
| Core Model | 🟡 In Progress |
| Training Loop | ✅ Complete |
| Auto Releases | ✅ Complete |
| Dataset Cycling | ✅ Complete |
| Error Recovery | ✅ Complete |
| Progress Tracking | ✅ Complete |
| Chat Interface | 🟡 In Progress |
| API | 🔴 Planned |

<br>

---

<br>

## 📄 License

MIT License - see [LICENSE](LICENSE)

<br>

---

<br>

**Built with 🤖 by [MeridianAlgo](https://github.com/MeridianAlgo)**

*Continuously training on GitHub Actions*

</div>
