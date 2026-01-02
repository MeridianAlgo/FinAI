"""Generate 25 QA responses from local model (CPU). Saves outputs to outputs/generation_qa.txt

This script will attempt to load a checkpoint at `checkpoints/checkpoint-50.pt` if present
and will fall back to a freshly-initialized model. It will try to load state dict with
`strict=False` and report missing/unexpected keys.
"""
import os
import torch
from transformers import AutoTokenizer
from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel


QUESTIONS = [
    "What is the definition of market volatility?",
    "Explain in one paragraph what value investing is.",
    "How would you hedge a portfolio against interest rate risk?",
    "Summarize the steps to backtest a trading strategy.",
    "What are common pitfalls when training language models on financial data?",
    "Give a short example of a prompt for generating earnings-call summaries.",
    "What does Sharpe ratio measure?",
    "Explain overfitting in machine learning and how to detect it.",
    "List three risk metrics used in portfolio management.",
    "How do you preprocess raw financial text for an LM?",
    "What is tokenization and why is it important for NLP models?",
    "How to choose sequence length for training a transformer?",
    "Describe a lightweight checkpointing strategy for frequent training.",
    "What are parameter-efficient fine-tuning methods?",
    "How can you speed up training on CPU-only machines?",
    "What is the role of learning rate schedulers?",
    "How should one validate a generative financial model before production?",
    "Write a short compliance-safe prompt to generate investment advice.",
    "Explain why mixed-precision training is useful and whether it applies to CPUs.",
    "How to evaluate model drift over repeated daily training runs?",
    "Give a one-paragraph checklist before pushing a model to Hugging Face Hub.",
    "What monitoring metrics should run during each training hour?",
    "How to reduce tokenization overhead when streaming datasets?",
    "Describe pros/cons of training from scratch vs fine-tuning a base model.",
    "What are recommended defaults for batch size and learning rate for small models?",
]


def try_load_checkpoint(model, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        return False, "no checkpoint"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        info = f"loaded with missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        return True, info
    except Exception as e:
        return False, f"load error: {e}"


def simple_detokenize(token_ids, cfg_vocab):
    # Map token ids to printable ASCII as a readable stand-in
    return ''.join(chr((t % 95) + 32) for t in token_ids)


def inspect_checkpoint_meta(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    meta = {}
    if "transformer.transformer.wte.weight" in state:
        meta["vocab_size"] = state["transformer.transformer.wte.weight"].shape[0]
        meta["embed_dim"] = state["transformer.transformer.wte.weight"].shape[1]
    if "transformer.transformer.wpe.weight" in state:
        meta["max_seq_len"] = state["transformer.transformer.wpe.weight"].shape[0]
    # estimate n_layers
    layers = set()
    for k in state.keys():
        if k.startswith("transformer.transformer.h."):
            parts = k.split('.')
            if len(parts) > 3:
                try:
                    layers.add(int(parts[3]))
                except:
                    pass
    if layers:
        meta["n_layers"] = max(layers) + 1
    return meta


def main():
    # Prefer model directory if available (HF-style saved model)
    model_dir = os.path.join("checkpoints", "model")
    if os.path.exists(model_dir):
        print(f"Loading model from {model_dir}")
        model = FinAIModel.from_pretrained(model_dir, device="cpu")
        cfg = model.config
        loaded = True
        info = f"loaded from {model_dir}"
    else:
        ckpt_path = os.path.join("checkpoints", "checkpoint-50.pt")
        meta = inspect_checkpoint_meta(ckpt_path)

        if meta:
            # build config matching checkpoint
            vocab_size = int(meta.get("vocab_size", 50257))
            embed_dim = int(meta.get("embed_dim", 384))
            n_layers = int(meta.get("n_layers", 6))
            print(f"Found checkpoint meta: vocab={vocab_size}, embed_dim={embed_dim}, n_layers={n_layers}")
            cfg = FinAIConfig(vocab_size=vocab_size, embed_dim=embed_dim, n_layers=n_layers, n_heads=6, ff_dim=embed_dim * 4, max_seq_len=int(meta.get("max_seq_len", 512)))
        else:
            cfg = FinAIConfig.from_preset("tiny", vocab_size=128, max_seq_len=128)

        model = FinAIModel(cfg)
        model.eval()

        loaded, info = try_load_checkpoint(model, ckpt_path)

    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generation_qa.txt")

    # Try to load a GPT-2 tokenizer to decode outputs into readable text. This may download tokenizer files.
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        tokenizer = None

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint_loaded: {loaded} - {info}\n\n")

        for i, q in enumerate(QUESTIONS, start=1):
            # Encode prompt using tokenizer if available, else fallback to char-mapped ids
            if tokenizer is not None:
                enc = tokenizer.encode(q, add_special_tokens=False, truncation=True, max_length=cfg.max_seq_len)
                input_ids = torch.tensor([enc], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
            else:
                token_ids = [ord(c) % cfg.vocab_size for c in q][:cfg.max_seq_len]
                input_ids = torch.tensor([token_ids], dtype=torch.long)
                attention_mask = None

            # Generate with improved decoding defaults to avoid repetition
            try:
                gen_kwargs = dict(
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
                # add attention mask and pad/eos ids if tokenizer available
                if tokenizer is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                    if tokenizer.pad_token_id is not None:
                        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
                    if tokenizer.eos_token_id is not None:
                        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

                out = model.generate(input_ids, **gen_kwargs)
                gen_ids = out[0].tolist()
                if tokenizer is not None:
                    gen_text = tokenizer.decode(gen_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                else:
                    gen_text = simple_detokenize(gen_ids, cfg.vocab_size)
            except Exception as e:
                gen_text = f"<generation_error: {e}>"

            f.write(f"Q{i}: {q}\n")
            f.write(f"A{i}: {gen_text}\n\n")

    print(f"Wrote {len(QUESTIONS)} Q/A to {out_path}. checkpoint_loaded={loaded} info={info}")


if __name__ == "__main__":
    main()
