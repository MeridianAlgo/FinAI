# Deployment Guide

This document explains how to build and run Fin.AI in a container for production-like environments.

## Build Docker Image

```bash
# Build image
docker build -t meridianalgo/finai:latest .

# Run container (local inference)
docker run --rm -it -v $(pwd)/checkpoints:/app/checkpoints -e HF_TOKEN=$HF_TOKEN meridianalgo/finai:latest
```

## Security & Secrets

- Never hardcode `HF_TOKEN` or `WANDB_API_KEY` in the image or repo.
- Use Docker secrets or environment variables injected by your orchestration platform.

## Scaling Notes

- Consider using a model server (TorchServe, FastAPI, or custom Flask app) for production inference.
- Keep model artifacts in a shared volume or object storage (S3, GCS).
