#!/usr/bin/env bash
set -euo pipefail

# Basic system deps
sudo apt update
sudo apt install -y python3 python3-pip git ca-certificates

# Python packages
pip3 install --upgrade pip
pip3 install feedparser pandas

echo "[OK] Dependencies installed."
