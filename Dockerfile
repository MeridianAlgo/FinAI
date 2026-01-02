# Multi-stage Dockerfile for Fin.AI
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN useradd -m finai && chown -R finai:finai /app
USER finai

ENV MODEL_DIR=/app/checkpoints/model
VOLUME ["/data", "/app/checkpoints"]

CMD ["python", "generate.py"]
