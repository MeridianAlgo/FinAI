#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/pi/FinAI"              # Adjust if different
PY="/usr/bin/python3"
BASH="/bin/bash"

COLLECT="$REPO_DIR/scripts/pi_news/collect_news.py"
PROCESS="$REPO_DIR/scripts/pi_news/process_data.py"
GITPUSH="$REPO_DIR/scripts/pi_news/git_commit_push.sh"
LOG_DIR="$REPO_DIR"

# Ensure executable flags
chmod +x "$COLLECT" "$PROCESS" "$GITPUSH"

# Current crontab (if any)
CURRENT=$(crontab -l 2>/dev/null || true)

ADD_COLLECT="0 7-19 * * * $PY $COLLECT >> $LOG_DIR/log_collect.log 2>&1"
ADD_PROCESS="0 19 * * * $PY $PROCESS >> $LOG_DIR/log_process.log 2>&1"
ADD_PUSH="5 19 * * * $BASH $GITPUSH >> $LOG_DIR/log_git.log 2>&1"

NEW_CRON="$CURRENT"

if ! echo "$CURRENT" | grep -Fq "$COLLECT"; then
  NEW_CRON="$NEW_CRON
$ADD_COLLECT"
fi
if ! echo "$CURRENT" | grep -Fq "$PROCESS"; then
  NEW_CRON="$NEW_CRON
$ADD_PROCESS"
fi
if ! echo "$CURRENT" | grep -Fq "$GITPUSH"; then
  NEW_CRON="$NEW_CRON
$ADD_PUSH"
fi

echo "$NEW_CRON" | crontab -

echo "[OK] Cron installed. Use: crontab -l"
