import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_id = os.getenv("HF_MODEL_ID", "meridianal/FinAI")
    print(f"[INFO] Download/load: {model_id}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "hidden_act") and config.hidden_act == "swiglu":
        config.hidden_act = "silu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    prompts = [
        "### Instruction:\nCalculate the compound interest on $10,000 at 5% over 10 years.\n\n### Response:\n",
        "### Instruction:\nExplain what a P/E ratio is and how it is used.\n\n### Response:\n",
        "### Instruction:\nGive a quick summary of what an ETF is.\n\n### Response:\n",
    ]

    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "96"))
    min_new_tokens = int(os.getenv("MIN_NEW_TOKENS", "32"))
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    top_p = float(os.getenv("TOP_P", "0.9"))

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    for i, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **gen_kwargs,
            )
        output_len = int(out.shape[-1])
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        text_with_special = tokenizer.decode(out[0], skip_special_tokens=False)

        print("\n" + "=" * 80)
        print(f"PROMPT {i}:")
        print(prompt)
        print("OUTPUT:")
        continuation = text[len(prompt) :] if text.startswith(prompt) else text
        print(continuation)
        print(
            f"[DEBUG] input_len={input_len} output_len={output_len} new_tokens={output_len - input_len}"
        )
        if output_len > input_len:
            print(f"[DEBUG] generated_token_ids_tail={out[0, input_len: input_len + 16].tolist()}")
        print(f"[DEBUG] decoded_with_special_repr={text_with_special!r}")
        print(f"[DEBUG] continuation_len={len(continuation)}")
        print(f"[DEBUG] continuation_repr={continuation!r}")


if __name__ == "__main__":
    main()
