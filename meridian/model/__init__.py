"""Meridian.AI model helpers.

Real training uses Qwen2.5-0.5B via ``AutoModelForCausalLM`` in ``train.py``.
This module only provides a tiny model for smoke/unit tests so we don't ship a
whole custom architecture just to exercise the trainer plumbing.
"""

from transformers import Qwen2Config, Qwen2ForCausalLM


def build_smoke_model(
    vocab_size: int = 4096,
    hidden_size: int = 128,
    num_layers: int = 4,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    intermediate_size: int = 352,
    max_position_embeddings: int = 256,
    tie_word_embeddings: bool = True,
) -> Qwen2ForCausalLM:
    """Small randomly-initialized Qwen2 model — same family as the production base."""
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
    )
    return Qwen2ForCausalLM(cfg)


__all__ = ["build_smoke_model"]
