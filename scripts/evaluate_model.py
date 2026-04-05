"""Meridian.AI - Model Evaluation Script.

Tests the fine-tuned checkpoint against the base SmolLM2-360M model.
Measures:
  1. Perplexity on finance-specific text
  2. Generation quality on finance prompts
  3. Basic coherence / repetition checks
"""

import sys
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Finance evaluation prompts ──────────────────────────────────────────
GENERATION_PROMPTS = [
    "What is the current P/E ratio of Apple and what does it indicate?",
    "Explain the difference between a stock and a bond.",
    "Calculate compound interest on $10,000 at 5% annual rate over 10 years.",
    "What is the Black-Scholes option pricing model?",
    "What are ETFs and how do they differ from mutual funds?",
    "Explain the concept of dollar-cost averaging in investing.",
    "What is a yield curve inversion and why does it matter?",
    "How does the Federal Reserve influence interest rates?",
]

# Finance-specific text for perplexity evaluation
PERPLEXITY_TEXTS = [
    "The price-to-earnings ratio is a fundamental valuation metric used by investors to determine whether a stock is overvalued or undervalued relative to its earnings. A high P/E ratio may indicate that the market expects future earnings growth.",
    "Compound interest is interest calculated on the initial principal and also on the accumulated interest from previous periods. The formula is A = P(1 + r/n)^(nt), where P is principal, r is annual rate, n is compounding frequency, and t is time.",
    "The Federal Reserve uses monetary policy tools including the federal funds rate, open market operations, and reserve requirements to influence economic conditions. When the Fed raises rates, borrowing becomes more expensive, which can slow inflation.",
    "An exchange-traded fund (ETF) is a type of pooled investment security that operates much like a mutual fund. Typically, ETFs track a particular index, sector, commodity, or other assets, but unlike mutual funds, ETFs can be purchased or sold on a stock exchange.",
    "The yield curve shows the relationship between bond yields and maturities. An inverted yield curve, where short-term rates exceed long-term rates, has historically been a reliable predictor of economic recessions.",
]


def load_model(model_path, model_name="model"):
    """Load a model and tokenizer from a path or HuggingFace ID."""
    print(f"\n{'-'*60}")
    print(f"  Loading {model_name}: {model_path}")
    t0 = time.time()

    # Try loading tokenizer from the checkpoint path first, fallback to HF
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"  [WARN] Tokenizer load from {model_path} failed: {e}")
        # Fall back to the known base tokenizer
        fallback = "HuggingFaceTB/SmolLM2-360M"
        print(f"  [INFO] Falling back to tokenizer from {fallback}")
        tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    elapsed = time.time() - t0
    print(f"  Parameters: {total_params:,}")
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"{'-'*60}")
    return model, tokenizer


def compute_perplexity(model, tokenizer, texts):
    """Compute perplexity on a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

        num_tokens = input_ids.shape[1] - 1  # labels are shifted
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def check_repetition(text, n=3):
    """Check for n-gram repetition rate."""
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0

    unique = set(ngrams)
    repetition_rate = 1.0 - len(unique) / len(ngrams)
    return repetition_rate


def generate_text(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7):
    """Generate text from a prompt."""
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            min_new_tokens=16,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids.shape[1] - input_len
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the response portion
    if "### Response:" in full_text:
        response = full_text.split("### Response:")[-1].strip()
    else:
        response = full_text[len(formatted):].strip() if full_text.startswith(formatted) else full_text

    tokens_per_sec = new_tokens / gen_time if gen_time > 0 else 0
    return response, new_tokens, gen_time, tokens_per_sec


def evaluate_model(model, tokenizer, model_name="Model"):
    """Run full evaluation on a model."""
    results = {"name": model_name, "prompts": [], "perplexity": None}

    # ── 1. Perplexity ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PERPLEXITY EVALUATION - {model_name}")
    print(f"{'='*60}")

    ppl, avg_loss = compute_perplexity(model, tokenizer, PERPLEXITY_TEXTS)
    results["perplexity"] = ppl
    results["avg_loss"] = avg_loss
    print(f"  Average Loss:  {avg_loss:.4f}")
    print(f"  Perplexity:    {ppl:.2f}")

    # ── 2. Generation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GENERATION EVALUATION - {model_name}")
    print(f"{'='*60}")

    total_rep = 0.0
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(GENERATION_PROMPTS, 1):
        response, new_tokens, gen_time, tok_per_sec = generate_text(model, tokenizer, prompt)
        rep_rate = check_repetition(response)

        total_rep += rep_rate
        total_tokens += new_tokens
        total_time += gen_time

        result = {
            "prompt": prompt,
            "response": response,
            "new_tokens": new_tokens,
            "gen_time_s": round(gen_time, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "repetition_rate": round(rep_rate, 3),
        }
        results["prompts"].append(result)

        print(f"\n  +-- Prompt {i}/{len(GENERATION_PROMPTS)} -----------------------")
        print(f"  | Q: {prompt}")
        print(f"  | A: {response[:300]}{'...' if len(response) > 300 else ''}")
        print(f"  | Tokens: {new_tokens} | Time: {gen_time:.1f}s | {tok_per_sec:.1f} tok/s | Rep: {rep_rate:.1%}")
        print(f"  +{'-'*50}")

    avg_rep = total_rep / len(GENERATION_PROMPTS) if GENERATION_PROMPTS else 0
    avg_tps = total_tokens / total_time if total_time > 0 else 0

    results["avg_repetition_rate"] = avg_rep
    results["avg_tokens_per_sec"] = avg_tps
    results["total_generation_time_s"] = round(total_time, 2)

    print(f"\n  Summary:")
    print(f"    Avg Repetition Rate: {avg_rep:.1%}")
    print(f"    Avg Tokens/sec:      {avg_tps:.1f}")
    print(f"    Total Gen Time:      {total_time:.1f}s")

    return results


def main():
    print("=" * 60)
    print("  MERIDIAN.AI - MODEL EVALUATION")
    print("  Comparing fine-tuned checkpoint vs base model")
    print("=" * 60)

    checkpoint_path = os.getenv("CHECKPOINT_PATH", "./hf_model/checkpoint")
    base_model_id = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M")
    skip_base = os.getenv("SKIP_BASE", "0") == "1"

    all_results = {}

    # ── Evaluate fine-tuned model ─────────────────────────────────────
    print("\n\n" + "#" * 60)
    print("  EVALUATING: FINE-TUNED CHECKPOINT")
    print("#" * 60)

    ft_model, ft_tokenizer = load_model(checkpoint_path, "Fine-tuned Meridian")
    ft_results = evaluate_model(ft_model, ft_tokenizer, "Fine-tuned Meridian")
    all_results["fine_tuned"] = ft_results

    # Free memory before loading base model
    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # ── Evaluate base model ───────────────────────────────────────────
    if not skip_base:
        print("\n\n" + "#" * 60)
        print("  EVALUATING: BASE MODEL (SmolLM2-360M)")
        print("#" * 60)

        try:
            base_model, base_tokenizer = load_model(base_model_id, "Base SmolLM2-360M")
            base_results = evaluate_model(base_model, base_tokenizer, "Base SmolLM2-360M")
            all_results["base"] = base_results
            del base_model
            gc.collect()
        except Exception as e:
            print(f"\n  [WARN] Could not load base model: {e}")
            print("  Skipping base model comparison.")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)

    headers = ["Metric", "Fine-tuned"]
    has_base = "base" in all_results
    if has_base:
        headers.append("Base")
        headers.append("Delta")

    # Table
    ft = all_results["fine_tuned"]
    rows = [
        ("Perplexity", f"{ft['perplexity']:.2f}"),
        ("Avg Loss", f"{ft['avg_loss']:.4f}"),
        ("Avg Repetition", f"{ft['avg_repetition_rate']*100:.1f}%"),
        ("Avg Tok/s", f"{ft['avg_tokens_per_sec']:.1f}"),
    ]

    if has_base:
        base = all_results["base"]
        extended_rows = []
        base_vals = [base["perplexity"], base["avg_loss"], base["avg_repetition_rate"], base["avg_tokens_per_sec"]]
        ft_vals = [ft["perplexity"], ft["avg_loss"], ft["avg_repetition_rate"], ft["avg_tokens_per_sec"]]
        for (name, ft_str), bv, fv in zip(rows, base_vals, ft_vals):  # noqa: B007
            delta = fv - bv
            direction = "v" if delta < 0 else "^" if delta > 0 else "="
            # For perplexity and loss, lower is better
            if name in ("Perplexity", "Avg Loss", "Avg Repetition"):  # lower is better
                quality = "OK" if delta < 0 else "BAD" if delta > 0 else "="
            else:
                quality = "OK" if delta > 0 else "BAD" if delta < 0 else "="

            if name == "Avg Repetition":  # format as percentage
                extended_rows.append((name, ft_str, f"{bv*100:.1f}%", f"{delta*100:+.1f}% {direction} {quality}"))
            elif name == "Avg Tok/s":
                extended_rows.append((name, ft_str, f"{bv:.1f}", f"{delta:+.1f} {direction} {quality}"))
            else:
                extended_rows.append((name, ft_str, f"{bv:.2f}" if name == "Perplexity" else f"{bv:.4f}", f"{delta:+.4f} {direction} {quality}"))
        rows = extended_rows

    # Print table
    print(f"\n  {'Metric':<20} {'Fine-tuned':>12}", end="")
    if has_base:
        print(f" {'Base':>12} {'Delta':>18}", end="")
    print()
    print(f"  {'-'*20} {'-'*12}", end="")
    if has_base:
        print(f" {'-'*12} {'-'*18}", end="")
    print()

    for row in rows:
        print(f"  {row[0]:<20} {row[1]:>12}", end="")
        if has_base and len(row) > 2:
            print(f" {row[2]:>12} {row[3]:>18}", end="")
        print()

    # Save results
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
