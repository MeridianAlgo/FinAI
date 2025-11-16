#!/bin/bash

echo "Auto-pushing to GitHub..."
echo

git add .
git commit -m "Auto-push: $(date)"
git push origin main --force

echo
echo "Push completed!"
