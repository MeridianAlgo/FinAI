#!/usr/bin/env python3
import os
import time
import json
import shutil
import tempfile
import urllib.request
import urllib.error
from urllib.parse import urljoin, urlencode
import traceback

import random
import numpy as np
import torch

from datasets import load_dataset
from src.core.finai import FinAI

# Always use CPU on workers
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Global headers (set in main from --token)
HEADERS = {}


def http_get_json(url, timeout=60):
    req = urllib.request.Request(url, headers=HEADERS, method='GET')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def http_post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    headers.update(HEADERS)
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def http_post_binary(url, data: bytes, timeout=600):
    headers = {'Content-Type': 'application/octet-stream'}
    headers.update(HEADERS)
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def extract_text_from_dataset(dataset, split="train"):
    texts = []
    try:
        if split and split in dataset:
            data = dataset[split]
        else:
            # fallback to first available split
            data = dataset[list(dataset.keys())[0]]

        print(f"  Processing {len(data)} examples...")
        text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query', 'answer', 'response']

        for item in data:
            text = None
            for field in text_fields:
                if field in item and item[field]:
                    text = item[field]
                    if isinstance(text, str):
                        break
            if not text or not isinstance(text, str):
                for _, value in item.items():
                    if isinstance(value, str) and value.strip():
                        if not text or len(value) > len(text):
                            text = value
                if not text or len(text) < 10:
                    text = " ".join([str(v) for v in item.values() if isinstance(v, (str, int, float)) and str(v).strip()])
            if text and isinstance(text, str) and len(text.strip()) > 10:
                texts.append(text.strip())
    except Exception as e:
        print(f"  WARNING: Error processing dataset: {e}")
    return texts


def extract_sharded_texts(dataset, split: str, shard_index: int, total_shards: int):
    texts = []
    num_tokens = 0
    try:
        if split and split in dataset:
            data = dataset[split]
        else:
            # fallback to first available split
            data = dataset[list(dataset.keys())[0]]

        print(f"  Processing {len(data)} examples for shard {shard_index}/{total_shards}...")
        text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query', 'answer', 'response']
        for i, item in enumerate(data):
            if (i % total_shards) != shard_index:
                continue
            text = None
            for field in text_fields:
                if field in item and item[field]:
                    text = item[field]
                    if isinstance(text, str):
                        break
            if not text or not isinstance(text, str):
                for _, value in item.items():
                    if isinstance(value, str) and value.strip():
                        if not text or len(value) > len(text):
                            text = value
                if not text or len(text) < 10:
                    text = " ".join([str(v) for v in item.values() if isinstance(v, (str, int, float)) and str(v).strip()])
            if text and isinstance(text, str) and len(text.strip()) > 10:
                t = text.strip()
                texts.append(t)
                num_tokens += (len(t.encode('utf-8')) + 2)
    except Exception as e:
        print(f"  WARNING: Error processing dataset shard: {e}")
    return texts, num_tokens


def safe_slug(name: str) -> str:
    return name.replace('/', '_').replace('\\', '_').replace(':', '_')


def train_job(job: dict):
    name = job.get('name')
    config = job.get('config') or None
    split = job.get('split') or 'train'
    shard_index = int(job.get('shard_index', -1))
    total_shards = int(job.get('total_shards', 1))
    seed = int(job.get('seed', 42))

    print(f"=== Starting job: {name} (split={split}, config={config}) ===")
    if shard_index >= 0 and total_shards > 1:
        print(f"Shard {shard_index+1}/{total_shards}")

    # Deterministic init across workers
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading dataset: {name}")
    dataset = load_dataset(name, config) if config else load_dataset(name)

    print("Extracting text from dataset...")
    if shard_index >= 0 and total_shards > 1:
        texts, num_tokens = extract_sharded_texts(dataset, split, shard_index, total_shards)
    else:
        texts = extract_text_from_dataset(dataset, split)
        num_tokens = sum((len(t.encode('utf-8')) + 2) for t in texts)

    if not texts:
        print("  WARNING: No text data found for this shard")
        return False, "No text data found", shard_index, total_shards, 0
    print(f"  Extracted {len(texts)} text samples (approx {num_tokens} bytes)")

    # Write to a temporary file for FinAI.train_from_file
    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8', suffix='.txt') as tf:
        for line in texts:
            tf.write(line.replace('\n', ' ') + '\n')
        tmp_path = tf.name

    try:
        print("Initializing FinAI (CPU only)...")
        model = FinAI()
        model.train_from_file(tmp_path, use_gpu=False)

        # Copy saved model artifacts to a dataset-specific folder
        base_models = 'models'
        src_model = os.path.join(base_models, 'finai_gpt.pt')
        src_token = os.path.join(base_models, 'tokenizer.pkl')
        out_dir = os.path.join(base_models, 'distributed', safe_slug(name))
        os.makedirs(out_dir, exist_ok=True)
        if os.path.exists(src_model):
            shutil.copy2(src_model, os.path.join(out_dir, 'finai_gpt.pt'))
        if os.path.exists(src_token):
            shutil.copy2(src_token, os.path.join(out_dir, 'tokenizer.pkl'))

        print(f"Saved artifacts to {out_dir}")
        return True, out_dir, shard_index, total_shards, num_tokens
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def worker_loop(server_url: str, poll_seconds: int = 5, worker_id: str = ""):
    base_next = urljoin(server_url, '/next_job')
    # Include worker_id heartbeat as query string
    if worker_id:
        next_job_url = base_next + '?' + urlencode({'worker_id': worker_id})
    else:
        next_job_url = base_next
    report_url = urljoin(server_url, '/report_result')

    print(f"Worker connected to {server_url}. Polling every {poll_seconds}s. CPU-only mode.")

    while True:
        try:
            # Poll with heartbeat
            resp = http_get_json(next_job_url)
            job = resp.get('job') if isinstance(resp, dict) else None
            if not job:
                time.sleep(poll_seconds)
                continue

            try:
                success, result, shard_index, total_shards, num_tokens = train_job(job)
                if 'shard_index' in job:
                    # Upload shard checkpoint for aggregation
                    try:
                        model_file = os.path.join(result, 'finai_gpt.pt') if success else None
                        if success and model_file and os.path.exists(model_file):
                            params = {
                                'name': job.get('name'),
                                'shard_index': shard_index,
                                'total_shards': total_shards,
                                'num_samples': num_tokens,
                            }
                            url = urljoin(server_url, '/report_shard') + '?' + urlencode(params)
                            with open(model_file, 'rb') as f:
                                http_post_binary(url, f.read(), timeout=600)
                            print(f"Reported shard {shard_index+1}/{total_shards} for {job.get('name')}")
                        else:
                            # Fallback: mark dataset failed
                            http_post_json(report_url, {'name': job.get('name'), 'success': False, 'error': 'Training failed or model file missing', 'model_path': ''})
                            print("Reported failure due to missing model file")
                            break
                    except Exception as e:
                        err = f"Shard upload error: {str(e)}\n" + traceback.format_exc()
                        print(err)
                        http_post_json(report_url, {'name': job.get('name'), 'success': False, 'error': err, 'model_path': ''})
                        break
                else:
                    payload = {
                        'name': job.get('name'),
                        'success': bool(success),
                        'model_path': result if success else ''
                    }
                    if not success:
                        payload['error'] = str(result)
                    http_post_json(report_url, payload)
                    print(f"Reported result for {job.get('name')}: {'success' if success else 'failed'}")
                    if not success:
                        break
            except Exception as e:
                err = f"Worker training error: {str(e)}\n" + traceback.format_exc()
                print(err)
                try:
                    http_post_json(report_url, {'name': job.get('name'), 'success': False, 'error': err, 'model_path': ''})
                except Exception:
                    pass
                break
        except KeyboardInterrupt:
            print("\nWorker stopped by user")
            break
        except Exception as e:
            print(f"Worker error: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FinAI Distributed Worker (CPU only)')
    parser.add_argument('--server', required=True, help='Base URL of the distributed server, e.g., http://192.168.1.10:8000')
    parser.add_argument('--poll', type=int, default=5, help='Polling interval seconds')
    parser.add_argument('--id', default='', help='Worker ID for heartbeats and identification')
    parser.add_argument('--token', default=os.environ.get('FINAI_DISTRIB_TOKEN', ''), help='Auth token (also read from FINAI_DISTRIB_TOKEN)')
    args = parser.parse_args()

    # Set global headers
    if args.token:
        HEADERS['X-Auth-Token'] = args.token

    worker_loop(args.server, args.poll, args.id)
