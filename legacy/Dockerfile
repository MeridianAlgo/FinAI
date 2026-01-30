FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m finai && \
    chown -R finai:finai /app && \
    mkdir -p /app/checkpoints && \
    chown -R finai:finai /app/checkpoints

USER finai

ENV MODEL_DIR=/app/checkpoints/model

VOLUME ["/app/checkpoints"]

EXPOSE 8000

CMD ["python", "train.py", "--config", "config/model_config.yaml", "--datasets", "config/datasets.yaml"]
