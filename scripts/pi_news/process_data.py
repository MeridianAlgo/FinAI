#!/usr/bin/env python3
import pandas as pd
from datetime import datetime
import os

# Resolve repo root based on this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
RAW_DIR = os.path.join(REPO_DIR, "datasets", "news", "raw")
PROC_DIR = os.path.join(REPO_DIR, "datasets", "news", "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def process_today():
    today = datetime.now().strftime("%Y-%m-%d")
    inp = os.path.join(RAW_DIR, f"{today}_news.csv")
    out = os.path.join(PROC_DIR, f"{today}_processed.txt")

    if not os.path.exists(inp):
        print(f"[INFO] No raw news file for {today} at {inp}")
        return None

    df = pd.read_csv(inp)
    # Deduplicate across typical identity columns if they exist
    keep_cols = [c for c in ["id", "link", "title"] if c in df.columns]
    if keep_cols:
        df = df.drop_duplicates(subset=keep_cols)

    lines = []
    for _, r in df.iterrows():
        t = str(r.get("title", "")).strip()
        s = str(r.get("summary", "")).strip()
        link = str(r.get("link", "")).strip()
        pub = str(r.get("published", "")).strip()
        src = str(r.get("source", "")).strip()
        if not t and not s:
            continue
        lines.append(
            f"TITLE: {t}\nSUMMARY: {s}\nPUBLISHED: {pub}\nSOURCE: {src}\nLINK: {link}\n\n"
        )

    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[OK] Processed {len(lines)} items -> {out}")
    return out


if __name__ == "__main__":
    process_today()
