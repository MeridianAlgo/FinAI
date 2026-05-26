"""Meridian.AI — Comprehensive Diagnostic & Test Script.

Downloads the latest checkpoint from HuggingFace, runs generation tests,
computes perplexity, and prints a full diagnostic report.
"""

import io
import sys

# Fix Windows encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import math
import os
import time

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

REPO_ID = "meridianal/FinAI"
TOKEN = os.getenv("huggingface_token") or os.getenv("HF_TOKEN")
DOWNLOAD_DIR = "./hf_model"

FINANCE_PROMPTS = [
    "Explain the difference between a bond's yield to maturity and its coupon rate.",
    "What is a P/E ratio and how is it used to value stocks?",
    "Calculate the compound interest on $10,000 at 5% annual rate over 10 years.",
    "What is the Black-Scholes model used for?",
    "Explain what an ETF is and how it differs from a mutual fund.",
    "What is dollar-cost averaging?",
    "What does an inverted yield curve signal about the economy?",
    "How does the Federal Reserve control inflation through interest rates?",
]

PERPLEXITY_TEXTS = [
    "The price-to-earnings ratio is a fundamental valuation metric. A high P/E ratio may indicate the market expects strong future earnings growth.",
    "Compound interest is calculated on the initial principal and accumulated interest. The formula is A = P(1 + r/n)^(nt).",
    "The Federal Reserve uses the federal funds rate and open market operations to influence economic conditions and control inflation.",
    "An exchange-traded fund tracks a particular index or sector and can be bought or sold on a stock exchange like a regular stock.",
    "An inverted yield curve, where short-term rates exceed long-term rates, has historically been a predictor of economic recessions.",
]


def download_model():
    print("\n" + "=" * 70)
    print("  STEP 1: DOWNLOADING MODEL FROM HUGGINGFACE")
    print("=" * 70)
    print(f"  Repo: {REPO_ID}")
    print(f"  Token: {'SET' if TOKEN else 'NOT SET (will try anonymous)'}")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(DOWNLOAD_DIR, "checkpoint")

    t0 = time.time()
    try:
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            local_dir=DOWNLOAD_DIR,
            token=TOKEN,
        )
        elapsed = time.time() - t0
        print(f"  [OK] Downloaded to {local_dir} in {elapsed:.1f}s")

        # List files
        for root, dirs, files in os.walk(DOWNLOAD_DIR):
            for fname in files:
                fpath = os.path.join(root, fname)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                rel = os.path.relpath(fpath, DOWNLOAD_DIR)
                print(f"  {rel:50s}  {size_mb:6.1f} MB")

        return checkpoint_dir if os.path.isdir(checkpoint_dir) else DOWNLOAD_DIR
    except Exception as e:
        print(f"  [FAIL] Download failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_model(model_path):
    print("\n" + "=" * 70)
    print("  STEP 2: LOADING MODEL")
    print("=" * 70)
    print(f"  Path: {model_path}")

    # Check what's in the directory
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        print(f"  Files: {files}")

    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  [OK] Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")
    except Exception as e:
        print(f"  [WARN] Tokenizer load failed: {e}")
        print("  Falling back to Qwen/Qwen2.5-0.5B tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        elapsed = time.time() - t0
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  [OK] Model loaded in {elapsed:.1f}s")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Parameters: {total_params:,}")
        if hasattr(model, "config"):
            print(f"  Config model_type: {getattr(model.config, 'model_type', 'unknown')}")
            if hasattr(model.config, "num_hidden_layers"):
                print(f"  Layers: {model.config.num_hidden_layers}")
            if hasattr(model.config, "hidden_size"):
                print(f"  Hidden size: {model.config.hidden_size}")
        return model, tokenizer
    except Exception as e:
        print(f"  [FAIL] Model load failed: {e}")
        import traceback

        traceback.print_exc()
        return None, tokenizer


def compute_perplexity(model, tokenizer, texts):
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        num_tokens = inputs["input_ids"].shape[1] - 1
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss), avg_loss


def check_repetition(text, n=3):
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def generate_response(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7):
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,
            do_sample=True,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    new_tokens = output_ids.shape[1] - input_len
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = (
            full_text[len(formatted) :].strip() if full_text.startswith(formatted) else full_text
        )

    return response, new_tokens, gen_time


def score_response(prompt, response):
    """Simple heuristic scoring of response quality."""
    issues = []
    score = 100

    if len(response.strip()) < 30:
        issues.append("TOO SHORT (under 30 chars)")
        score -= 40
    if len(response.strip()) > 10 and check_repetition(response) > 0.3:
        issues.append(f"HIGH REPETITION ({check_repetition(response):.0%})")
        score -= 30
    # Check if response is mostly gibberish (long sequences of same tokens)
    words = response.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append(f"LOW VOCABULARY DIVERSITY ({unique_ratio:.0%} unique words)")
            score -= 25
    # Check if response addresses the question at all (keywords)
    prompt_lower = prompt.lower()
    resp_lower = response.lower()
    relevant_keywords = {
        "compound interest": ["interest", "principal", "rate", "%", "formula"],
        "p/e ratio": ["earnings", "price", "ratio", "valuation", "stock"],
        "etf": ["fund", "index", "trade", "stock", "exchange"],
        "yield curve": ["yield", "rate", "bond", "recession", "short", "long"],
        "federal reserve": ["rate", "inflation", "monetary", "interest", "fed"],
        "bond": ["yield", "coupon", "maturity", "interest", "rate"],
        "dollar-cost": ["average", "invest", "price", "share"],
        "black-scholes": ["option", "price", "volatility", "model"],
    }
    for keyword, expected in relevant_keywords.items():
        if keyword in prompt_lower:
            if not any(k in resp_lower for k in expected):
                issues.append(f"OFF-TOPIC (no relevant {keyword} terms)")
                score -= 20
            break

    return max(0, score), issues


def main():
    print("=" * 70)
    print("  MERIDIAN.AI — 3-MONTH DIAGNOSTIC REPORT")
    print("  Date: 2026-05-26  |  Dataset items processed: 64,464")
    print("=" * 70)

    # 1. Download
    model_path = download_model()
    if model_path is None:
        print("\n[FATAL] Could not download model. Aborting.")
        return

    # 2. Load
    model, tokenizer = load_model(model_path)
    if model is None:
        print("\n[FATAL] Could not load model. Aborting.")
        return

    # 3. Perplexity
    print("\n" + "=" * 70)
    print("  STEP 3: PERPLEXITY ON FINANCE TEXT")
    print("=" * 70)
    ppl, avg_loss = compute_perplexity(model, tokenizer, PERPLEXITY_TEXTS)
    print(f"  Average Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"  Perplexity:                 {ppl:.2f}")
    if ppl < 10:
        print("  Quality:   EXCELLENT (ppl < 10)")
    elif ppl < 30:
        print("  Quality:   GOOD (ppl 10-30)")
    elif ppl < 100:
        print("  Quality:   POOR (ppl 30-100) — model struggles with finance text")
    else:
        print("  Quality:   VERY POOR (ppl > 100) — model has not learned finance well")

    # 4. Generation tests
    print("\n" + "=" * 70)
    print("  STEP 4: GENERATION QUALITY TESTS")
    print("=" * 70)

    total_score = 0
    all_issues = []
    responses = []
    for i, prompt in enumerate(FINANCE_PROMPTS, 1):
        response, n_tokens, gen_time = generate_response(model, tokenizer, prompt)
        score, issues = score_response(prompt, response)
        total_score += score
        all_issues.extend(issues)
        responses.append((prompt, response, score, issues))

        rep = check_repetition(response)
        print(f"\n  [{i}/{len(FINANCE_PROMPTS)}] Q: {prompt}")
        print(f"  A: {response[:300]}{'...' if len(response) > 300 else ''}")
        print(
            f"  Score: {score}/100 | Tokens: {n_tokens} | Time: {gen_time:.1f}s | Repetition: {rep:.0%}"
        )
        if issues:
            print(f"  ISSUES: {', '.join(issues)}")

    avg_score = total_score / len(FINANCE_PROMPTS)

    # 5. Verdict
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC VERDICT")
    print("=" * 70)
    print(f"  Perplexity:        {ppl:.1f}")
    print(f"  Avg Response Score: {avg_score:.1f}/100")
    print(f"  Total Issues Found: {len(all_issues)}")
    print()

    if avg_score >= 70 and ppl < 30:
        verdict = "GOOD — model has learned finance reasonably well"
    elif avg_score >= 40 or ppl < 100:
        verdict = "MEDIOCRE — model shows partial learning but responses are weak"
    else:
        verdict = "POOR — model responses are low quality; training needs significant fixes"

    print(f"  OVERALL: {verdict}")
    print()

    # Issue frequency
    if all_issues:
        from collections import Counter

        issue_counts = Counter(i.split("(")[0].strip() for i in all_issues)
        print("  Most common issues:")
        for issue, count in issue_counts.most_common(5):
            print(f"    - {issue}: {count}x")

    # 6. Architecture check
    print("\n" + "=" * 70)
    print("  STEP 5: ARCHITECTURE & TRAINING AUDIT")
    print("=" * 70)
    if hasattr(model, "config"):
        cfg = model.config
        print(f"  Model type:         {getattr(cfg, 'model_type', 'unknown')}")
        print(f"  Architecture class: {type(model).__name__}")
        is_qwen = "qwen2" in str(getattr(cfg, "model_type", "")).lower()
        is_custom = "meridian" in str(getattr(cfg, "model_type", "")).lower()
        if is_qwen:
            print("  [!] This is a standard Qwen2 model — the custom SMoE arch was NOT used")
        elif is_custom:
            print("  [OK] Custom Meridian SMoE architecture confirmed")
        else:
            print(f"  [?] Unknown architecture: {getattr(cfg, 'model_type', 'unknown')}")

    print("\n  Done. See CHANGELOG.md for full diagnosis and improvement plan.")


if __name__ == "__main__":
    main()
