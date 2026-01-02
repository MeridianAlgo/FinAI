#!/usr/bin/env python3
"""
Test all datasets to ensure they load correctly

Usage:
    python test_datasets.py
"""

import yaml
from datasets import load_dataset

"""Dataset scripts are moved to `legacy/` for manual execution.

This module is intentionally inert during pytest runs to avoid downloading
datasets automatically. Run datasets checks manually from `legacy/` when
needed.
"""

def main():
    print("Dataset tests were moved to legacy/ to avoid automatic downloads.")


if __name__ == "__main__":
    main()
