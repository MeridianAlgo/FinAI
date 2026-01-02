import torch

from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel


def set_seed(seed: int = 42):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detokenize_simple(token_ids):
    # Map small token ids to safe printable characters for display
    chars = []
    for t in token_ids:
        c = t % 95 + 32  # printable ASCII range 32-126
        chars.append(chr(c))
    return "".join(chars)


def test_greedy_generation_increases_length_and_tokens_in_vocab():
    set_seed(2026)

    cfg = FinAIConfig.from_preset("tiny", vocab_size=128, max_seq_len=32)
    model = FinAIModel(cfg)
    model.eval()

    prompt_len = 6
    # deterministic prompt: [1,2,3,4,5,6]
    input_ids = torch.tensor([list(range(1, prompt_len + 1))], dtype=torch.long)

    # Greedy generation - deterministic
    out = model.generate(input_ids, max_new_tokens=10, do_sample=False)

    assert out.shape[1] == prompt_len + 10, "Generated sequence length mismatch"

    # Ensure all tokens are within vocab
    assert torch.all((out >= 0) & (out < cfg.vocab_size)), "Token id out of range"

    # Basic smoke: detokenize and ensure we have a string
    decoded = detokenize_simple(out[0].tolist())
    assert isinstance(decoded, str) and len(decoded) > 0
