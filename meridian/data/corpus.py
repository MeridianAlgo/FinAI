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

import os
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

# Corrections to FinanceDataPipeline.DATASETS. The first corpus build revealed that 14 of
# its 27 entries yield zero documents — their configured column simply does not exist, so
# _format_text returns "" and every row is dropped. Since the trainer shares _format_text,
# those datasets have been contributing nothing to training either; see
# `scripts/validate_datasets.py`, which checks every spec against the live schema.
#
# Three distinct causes:
#   1. The column is `sentence`, not `text`. Also switches to `label_text`, which is already
#      a readable string, so the int->str label_map is unnecessary.
#   2. MTEB *retrieval* sets whose `default` config holds only qrels (query-id, corpus-id,
#      score). The documents live in the `corpus` config, which is where the finance text
#      actually is — TATQA's, for instance, is markdown tables of financial statements.
#   3. Sets published with only a `test` split.
SPEC_OVERRIDES: dict[str, dict] = {
    # 1. `sentence` column
    **{
        name: {"text_field": "sentence", "label_field": "label_text", "label_map": None}
        for name in (
            "FinanceMTEB/OpenFinDataSentiment",
            "FinanceMTEB/SemEva2017_Headline",
            "FinanceMTEB/FOMC",
            "FinanceMTEB/FinancialFraud",
            "FinanceMTEB/FinFE",
            "FinanceMTEB/FinEvaSentiment",
        )
    },
    # 2. retrieval sets — take the corpus documents as plain text
    **{
        name: {
            "config": "corpus",
            "split": "train",
            "text_field": "text",
            "instruction_field": None,
            "label_field": None,
            "label_map": None,
            "prompt_template": None,
        }
        for name in (
            "FinanceMTEB/FinQA",
            "FinanceMTEB/TATQA",
            "FinanceMTEB/TradeTheEventNews",
            "FinanceMTEB/TradeTheEventEncyclopedia",
        )
    },
    # 3. test-only split
    "FinanceMTEB/synthetic_pii_finance_en": {"split": "test", "text_field": "sentences"},
}

# Dropped rather than repaired.
#   - The three Chinese-language sets are noise for an English finance model; their combined
#     weight is ~2.4%, and that budget is better spent on English filings.
#   - Complaints exposes no readable schema on the Hub (no columns on its only split).
DROP_DATASETS = {
    "FinanceMTEB/AlphaFin",
    "FinanceMTEB/FinChinaSentiment",
    "FinanceMTEB/FinTruthQA",
    "FinanceMTEB/Complaints",
}


# --------------------------------------------------------------------------------------
# Sources added beyond the trainer's list, to fix the finance shortage measured in Phase 2:
# only 150.9M unique finance tokens existed, against a 500M-token corpus at 65% finance.
# --------------------------------------------------------------------------------------

# EDGAR-CORPUS (Loukas et al. 2021): 91,086 10-K filings, 1993-2020, ~5.7 GB of text, split
# by item. Public domain, and on its own ~10x the entire previous finance pool. These are the
# substantive prose sections; the ones left out (1B, 4, 9B, 10-14) are usually one-line
# boilerplate or cross-references to a proxy statement.
EDGAR_SECTIONS = (
    "section_1",  # Business
    "section_1A",  # Risk Factors
    "section_3",  # Legal Proceedings
    "section_5",  # Market for Registrant's Common Equity
    "section_7",  # MD&A — the richest financial reasoning in a filing
    "section_7A",  # Quantitative and Qualitative Disclosures About Market Risk
    "section_8",  # Financial Statements
    "section_9A",  # Controls and Procedures
)
EDGAR_MIN_SECTION_CHARS = 400

# MeridianAlgo/FinDB: scraped financial news, ~22.8k articles. Not an HF dataset, so it is
# read straight from the SQLite file in the repo.
FINDB_URL = "https://github.com/MeridianAlgo/FinDB/raw/main/financial_news.db"
# google_finance rows are unparsed Google News redirect URLs rather than article text —
# 11,387 of them, 0% usable by inspection. seeking_alpha is mostly one-line teasers.
FINDB_EXCLUDED_SOURCES = {"google_finance"}
FINDB_MIN_CHARS = 400


class _EosOnlyTokenizer:
    """``_format_text`` touches the tokenizer only for ``.eos_token``.

    Standing one of these up avoids loading a real tokenizer during corpus building —
    which matters because the corpus is what we train the tokenizer *on*.
    """

    def __init__(self, eos_token: str) -> None:
        self.eos_token = eos_token


def format_edgar(item: dict, eos_token: str) -> str:
    """Concatenate the substantive sections of one 10-K into a single document.

    Sections shorter than ``EDGAR_MIN_SECTION_CHARS`` are dropped: a filing that omits an
    item still emits its header plus "Not Applicable", and training on thousands of those
    teaches the model boilerplate rather than finance.
    """
    parts = [
        text.strip()
        for key in EDGAR_SECTIONS
        if len(text := (item.get(key) or "").strip()) >= EDGAR_MIN_SECTION_CHARS
    ]
    return "\n\n".join(parts) + eos_token if parts else ""


def resolve_parquet_urls(name: str, config: str, split: str) -> list[str]:
    """Find the Hub's auto-converted parquet files for a dataset.

    EDGAR-CORPUS is published as a loading *script*, and `datasets` 4.x dropped script
    support ("Dataset scripts are no longer supported"), so `load_dataset` cannot open it at
    all. The Hub separately converts every dataset to parquet on a `refs/convert/parquet`
    branch, and those files load natively — so we resolve them and read those instead.
    """
    import json
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode({"dataset": name, "config": config})
    url = f"https://datasets-server.huggingface.co/parquet?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    return [f["url"] for f in payload.get("parquet_files", []) if f.get("split") == split]


def iter_findb(eos_token: str, max_documents: int, cache_path: str = "findb.sqlite"):
    """Stream usable articles out of the FinDB SQLite database.

    Downloaded rather than streamed because it is a single 90 MB file in a git repo, not a
    Hub dataset.
    """
    import sqlite3
    import urllib.request

    if not os.path.exists(cache_path):
        print(f"    Downloading FinDB ({FINDB_URL}) ...", flush=True)
        urllib.request.urlretrieve(FINDB_URL, cache_path)

    connection = sqlite3.connect(cache_path)
    placeholders = ",".join("?" * len(FINDB_EXCLUDED_SOURCES))
    query = (
        "SELECT title, content FROM financial_news "
        f"WHERE source NOT IN ({placeholders}) AND LENGTH(content) >= ? "
        "AND COALESCE(is_duplicate, 0) = 0 ORDER BY published_date"
    )
    params = (*FINDB_EXCLUDED_SOURCES, FINDB_MIN_CHARS)

    emitted = 0
    for title, content in connection.execute(query, params):
        text = f"{title}\n\n{content}" if title else content
        # U+FFFD marks bytes already lost to a mis-decode upstream; leaving them in would
        # spend vocabulary on a character that carries nothing.
        text = text.replace("�", "").strip()
        # Guard against rows that are a URL blob rather than prose.
        if len(text) < FINDB_MIN_CHARS or text.count(" ") / len(text) < 0.10:
            continue
        yield text + eos_token
        emitted += 1
        if emitted >= max_documents:
            break
    connection.close()


# Weights are on the same scale as FinanceDataPipeline.DATASETS and renormalized with them.
# EDGAR is weighted to become the backbone of the finance side: it is the largest, cleanest,
# and most on-target finance text available, and it is public domain.
EXTRA_SOURCES: list[dict] = [
    {
        "name": "c3po-ai/edgar-corpus",
        "config": "full",
        "split": "train",
        "text_field": None,
        "formatter": "edgar",
        # Published as a loading script, which datasets 4.x cannot execute; read the Hub's
        # auto-converted parquet instead.
        "loader": "parquet_auto",
        "weight": 0.60,
        "heavy": True,
    },
    {
        "name": "MeridianAlgo/FinDB",
        "config": None,
        "split": "train",
        "text_field": None,
        "loader": "findb",
        "weight": 0.08,
    },
]


def domain_of(dataset_name: str) -> str:
    return "general" if dataset_name in GENERAL_DATASETS else "finance"


def slug(dataset_name: str) -> str:
    """Filesystem-safe stem for staging files."""
    return re.sub(r"[^A-Za-z0-9]+", "-", dataset_name).strip("-").lower()


def dataset_specs(include_heavy: bool = True, exclude: set[str] | None = None) -> list[dict]:
    """The dataset mix with corrections applied, normalized to sum to 1.0.

    Weights are renormalized *after* filtering, so dropping the Chinese sets redistributes
    their share across the rest rather than quietly shrinking the corpus.
    """
    exclude = (exclude or set()) | DROP_DATASETS
    specs = []
    for raw in list(FinanceDataPipeline.DATASETS) + EXTRA_SOURCES:
        name = raw["name"]
        if name in exclude or (not include_heavy and (name in HEAVY_DATASETS or raw.get("heavy"))):
            continue
        spec = dict(raw)
        spec.update(SPEC_OVERRIDES.get(name, {}))
        spec["domain"] = domain_of(name)
        specs.append(spec)

    total = sum(spec.get("weight", 0.0) for spec in specs) or 1.0
    for spec in specs:
        spec["weight"] = spec.get("weight", 0.0) / total
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
    if spec.get("loader") == "findb":
        try:
            yield from iter_findb(eos_token, max_documents)
        except Exception as exc:  # noqa: BLE001 — one unavailable source should not abort
            print(f"    [SKIP] {spec['name']}: {exc}", flush=True)
        return

    pipeline = FinanceDataPipeline.__new__(FinanceDataPipeline)
    pipeline.tokenizer = _EosOnlyTokenizer(eos_token)

    try:
        from datasets import load_dataset

        if spec.get("loader") == "parquet_auto":
            urls = resolve_parquet_urls(
                spec["name"], spec.get("config") or "default", spec["split"]
            )
            if not urls:
                raise RuntimeError("no auto-converted parquet files published")
            stream = load_dataset("parquet", data_files=urls, split="train", streaming=True)
        else:
            kwargs = {"path": spec["name"], "split": spec["split"], "streaming": True}
            if spec.get("config"):
                kwargs["name"] = spec["config"]
            stream = load_dataset(**kwargs)
    except Exception as exc:  # noqa: BLE001 — any Hub or schema failure should skip, not abort
        print(f"    [SKIP] {spec['name']}: {exc}", flush=True)
        return

    formatter = spec.get("formatter")
    emitted = 0
    chars = 0
    try:
        for item in stream:
            if formatter == "edgar":
                text = format_edgar(item, eos_token)
            else:
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
