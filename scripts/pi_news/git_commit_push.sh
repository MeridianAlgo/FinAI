#!/usr/bin/env bash
set -euo pipefail

# Load optional env (e.g., GITHUB_TOKEN) if present
if [[ -f "$HOME/.finai_env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.finai_env"
fi

REPO_DIR="/home/pi/FinAI"   # Absolute path on the Pi
BRANCH="master"              # Repo default branch

cd "$REPO_DIR"

# Optional: support HTTPS with token (no SSH). If GITHUB_TOKEN is set, use it.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  # Ensure remote uses token form for non-interactive cron pushes
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/MeridianAlgo/FinAI.git"
fi

# Stage only news dataset updates
git add datasets/news/raw/*.csv datasets/news/processed/*.txt 2>/dev/null || true

if ! git diff --cached --quiet; then
  git commit -m "Auto-update: $(date +'%Y-%m-%d %H:%M %Z')"
  git pull --rebase origin "$BRANCH"
  git push origin "$BRANCH"
else
  echo "[INFO] No changes to commit."
fi
