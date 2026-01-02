"""Simplified Fin.AI transformer model"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import json
import os

from fin_ai.model.config import FinAIConfig


class FinAIModel(nn.Module):
    """Simplified GPT-style transformer for Fin.AI"""

    def __init__(self, config: FinAIConfig):
        super().__init__()
        self.config = config

        # Use GPT2 architecture from transformers as backbone
        from transformers import GPT2Config, GPT2LMHeadModel

        gpt_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_seq_len,
            n_embd=config.embed_dim,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            n_inner=config.ff_dim,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.attention_dropout,
        )
        self.transformer = GPT2LMHeadModel(gpt_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        pad_token_id: int = None,
        eos_token_id: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """Use the underlying transformers' generate utility for richer decoding options.

        Supports repetition_penalty and no_repeat_ngram_size to reduce repetition.
        Additional kwargs are forwarded to `transformer.generate`.
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        gen_kwargs.update(kwargs)

        # Delegate to transformers' generation utilities which handle caching and decoding
        outputs = self.transformer.generate(input_ids=input_ids, **gen_kwargs)
        return outputs

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save our custom config
        with open(os.path.join(path, "finai_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        # Save underlying transformers model in HF format
        try:
            self.transformer.save_pretrained(path)
        except Exception:
            # Fallback to state_dict
            torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu"):
        # Load our custom config first if present
        cfg_path = os.path.join(path, "finai_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                config_dict = json.load(f)
            config = FinAIConfig(**config_dict)
        else:
            # Try fallback: if transformers-style config exists, we'll infer minimal settings
            config = FinAIConfig()

        model = cls(config)
        # Try to load HF-style model
        try:
            # Load via transformers
            model.transformer = model.transformer.from_pretrained(path)
            model.to(device)
            return model
        except Exception:
            # Fallback to state_dict
            state_path = os.path.join(path, "model.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=device)
                model.load_state_dict(state)
            model.to(device)
            return model

    def apply_lora(self, r: int = 8, alpha: int = 32, dropout: float = 0.05):
        """Apply LoRA adapters to the underlying transformers model if `peft` is installed.

        This is optional and will raise a helpful error if `peft` is not available.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise RuntimeError(
                "PEFT library not installed. Install `peft` to use LoRA."
            ) from e

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=dropout,
            bias="none",
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        return self
