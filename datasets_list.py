#!/usr/bin/env python3
"""
Dataset list for sequential training.
Add your datasets here, and train_sequential.py will train on each one individually.
"""

# Add all your datasets here in this format:
# Each entry is: ("dataset_name", "config_name" or None, "split_name" or None)

DATASETS = [
    ("FinGPT/fingpt-forecaster-dow30-202305-202405", None, None),
    ("Josephgflowers/Finance-Instruct-500k", None, None),
    ("sujet-ai/Sujet-Finance-Instruct-177k", None, None),
    ("virattt/financial-qa-10K", None, None),
]
