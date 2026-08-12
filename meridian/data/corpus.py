"""Corpus construction for MeridianLM pretraining (Phase 2).

Reuses the dataset specification in ``FinanceDataPipeline.DATASETS``: those repo IDs,
splits, field names, label maps, and prompt templates are already verified against the Hub
by the running trainer, so re-deriving them here would only invite drift. ``_format_text``
is reused for the same reason — the corpus should contain exactly the text the trainer
would have produced.

The one thing deliberately *not* reused is the streaming strategy. The trainer opens ~25
datasets concurrently with per-stream shuffle buffers, which is what drove RSS past 15 GB
(see the comments in ``train.yml``). Here we visit one dataset at a time, so peak memory is
one Arrow row-group rather than twenty-five, and mixing is done afterwards over tokenized
staging files where it costs nothing.
"""

from __future__ import annotations

import re
from typing import Iterator

from meridian.data.pipeline import FinanceDataPipeline

# Everything in the mix is finance except these. The split matters because Phase 3 tracks
# held-out perplexity separately per domain — a model improving on finance while rotting on
# general English is a specific failure we want to be able to see.
GENERAL_DATASETS = {
    "nvidia/OpenMathInstruct-2",
    "HuggingFaceFW/fineweb-edu",
    "yahma/alpaca-cleaned",
}

# fineweb-edu's row-groups are large enough to spike a 16 GB runner on their own; the
# trainer excludes it for that reason. We can afford it here only because we visit one
# dataset at a time, but it stays opt-in via --include-heavy.
HEAVY_DATASETS = {"HuggingFaceFW/fineweb-edu", "nvidia/OpenMathInstruct-2"}


class _EosOnlyTokenizer:
    """``_format_text`` touches the tokenizer only for ``.eos_token``.

    Standing one of these up avoids loading a real tokenizer during corpus building —
    which matters because the corpus is what we train the tokenizer *on*.
    """

    def __init__(self, eos_token: str) -> None:
        self.eos_token = eos_token


def domain_of(dataset_name: str) -> str:
    return "general" if dataset_name in GENERAL_DATASETS else "finance"


def slug(dataset_name: str) -> str:
    """Filesystem-safe stem for staging files."""
    return re.sub(r"[^A-Za-z0-9]+", "-", dataset_name).strip("-").lower()


def dataset_specs(include_heavy: bool = True, exclude: set[str] | None = None) -> list[dict]:
    """The dataset mix, normalized so weights sum to 1.0 over whatever survives filtering."""
    exclude = exclude or set()
    specs = [
        dict(spec)
        for spec in FinanceDataPipeline.DATASETS
        if spec["name"] not in exclude and (include_heavy or spec["name"] not in HEAVY_DATASETS)
    ]
    total = sum(spec.get("weight", 0.0) for spec in specs) or 1.0
    for spec in specs:
        spec["weight"] = spec.get("weight", 0.0) / total
        spec["domain"] = domain_of(spec["name"])
    return specs


def iter_documents(
    spec: dict,
    eos_token: str,
    max_documents: int,
    max_chars: int | None = None,
) -> Iterator[str]:
    """Stream formatted documents from one dataset.

    Yields at most ``max_documents`` non-empty strings. Failures are surfaced to the caller
    as an early return rather than an exception: one unavailable dataset should cost its
    slice of the mix, not the whole corpus build.
    """
    pipeline = FinanceDataPipeline.__new__(FinanceDataPipeline)
    pipeline.tokenizer = _EosOnlyTokenizer(eos_token)

    try:
        from datasets import load_dataset

        kwargs = {"path": spec["name"], "split": spec["split"], "streaming": True}
        if spec.get("config"):
            kwargs["name"] = spec["config"]
        stream = load_dataset(**kwargs)
    except Exception as exc:  # noqa: BLE001 — any Hub or schema failure should skip, not abort
        print(f"    [SKIP] {spec['name']}: {exc}", flush=True)
        return

    emitted = 0
    chars = 0
    try:
        for item in stream:
            text = pipeline._format_text(item, spec)
            if not text or not text.strip():
                continue
            yield text
            emitted += 1
            chars += len(text)
            if emitted >= max_documents:
                return
            if max_chars is not None and chars >= max_chars:
                return
    except Exception as exc:  # noqa: BLE001 — a mid-stream decode error should not be fatal
        print(f"    [WARN] {spec['name']} stopped after {emitted} docs: {exc}", flush=True)
