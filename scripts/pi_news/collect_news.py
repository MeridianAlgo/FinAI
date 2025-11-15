#!/usr/bin/env python3
import feedparser
import pandas as pd
from datetime import datetime
import os

# Resolve repo root based on this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
RAW_DIR = os.path.join(REPO_DIR, "datasets", "news", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Configure your RSS feeds here
FEEDS = [
    "http://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.investing.com/rss/news.rss",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://www.reuters.com/markets/us/rss"
]


def fetch_news():
    items = []
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:100]:  # cap per-feed to avoid huge files
                items.append({
                    "id": e.get("id") or e.get("link") or "",
                    "title": e.get("title", ""),
                    "summary": e.get("summary", e.get("description", "")),
                    "published": e.get("published", ""),
                    "link": e.get("link", ""),
                    "source": url
                })
        except Exception as ex:
            print(f"[WARN] {url}: {ex}")
    return items


def append_today(items):
    if not items:
        print("[INFO] No items fetched.")
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(RAW_DIR, f"{today}_news.csv")
    df = pd.DataFrame(items)
    # Deduplicate within this batch
    if not df.empty:
        keep_cols = [c for c in ["id", "link", "title"] if c in df.columns]
        if keep_cols:
            df = df.drop_duplicates(subset=keep_cols)
    # Append or create
    if os.path.exists(out):
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)
    return len(df)


if __name__ == "__main__":
    count = append_today(fetch_news())
    print(f"[OK] Appended {count} items to {RAW_DIR}")
