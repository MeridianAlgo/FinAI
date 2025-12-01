# FinAI: The Intelligent Financial Language Model

![FinAI Banner](https://img.shields.io/badge/FinAI-Smart%20%26%20Efficient-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active%20Training-success?style=for-the-badge)

**FinAI** is a specialized, high-efficiency Large Language Model (LLM) designed for financial analysis, sentiment detection, and market understanding. 

Built to be **"smart"** while remaining **cost-effective**, FinAI leverages a highly optimized ~14M parameter architecture ("Mini-GPT") that punches well above its weight class. It is trained daily on a diverse range of financial datasets to continuously improve its understanding of market dynamics.

## 🚀 Key Features

*   **Smart & Efficient**: ~14M parameters optimized for financial reasoning.
*   **Daily Continuous Learning**: Automatically trains on new data every day via GitHub Actions.
*   **Cost-Effective**: Designed to run and train on standard hardware (CPUs/Consumer GPUs).
*   **Diverse Knowledge Base**: Trained on news, tweets, SEC filings, and financial QA datasets.

## 📂 Project Structure

*   `src/`: Core source code for the model and training logic.
    *   `config.py`: Configuration for model architecture and training hyperparameters.
    *   `core/`: Model definitions.
*   `scripts/`: Utility scripts for data management and training.
    *   `train_daily_gh.py`: The script used by GitHub Actions for daily updates.
    *   `manage_datasets.py`: Tools for handling Hugging Face datasets.
*   `models/`: Stores the trained model checkpoints (`finai_gpt.pt`) and tokenizer.
*   `datasets/`: Local cache for datasets.
*   `.github/workflows/`: CI/CD pipelines for automated training.

## 🛠️ Getting Started

### Prerequisites

*   Python 3.10+
*   `pip`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/MeridianAlgo/FinAI.git
    cd FinAI
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**Training (Manual):**
To run a manual training session using the daily training script:
```bash
python scripts/train_daily_gh.py
```

**Inference (Coming Soon):**
Scripts to chat with FinAI are in development.

## 🤖 Automated Training

FinAI uses **GitHub Actions** to train itself daily.
1.  Every day at midnight, a workflow triggers.
2.  It selects a random financial dataset.
3.  It trains the model for a set number of steps.
4.  It pushes the updated model to a new branch and opens a Pull Request.
5.  **You merge the PR to save the knowledge!**

## 🤝 Contributing

We welcome contributions! Whether it's adding new datasets, optimizing the model, or fixing bugs.
Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 🔒 Security

For security concerns, please refer to our [SECURITY.md](SECURITY.md).

## 📊 Datasets

We use a curated list of high-quality financial datasets from Hugging Face, including:
*   `financial_phrasebank`
*   `finance-alpaca`
*   `twitter-financial-news-sentiment`
*   And many more!

---

<div align="center">
  <b>Made with love by MeridianAlgo</b>
</div>