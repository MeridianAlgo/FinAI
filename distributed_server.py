#!/usr/bin/env python3
import json
import os
import csv
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import traceback
import hashlib
import torch
import time

DATASETS_CSV = "datasets.csv"
TRAINED_CSV = "trained_datasets.csv"
CSV_HEADERS = ['name', 'config', 'split', 'date_trained', 'model_path', 'status']

LOCK = threading.Lock()
# Sessions track sharded training per dataset
SESSIONS = {}  # name -> { 'total_shards': int, 'assigned': set[int], 'completed': set[int], 'seed': int, 'split': str, 'config': str, 'upload_dir': str, 'shard_info': dict }
# Where incoming shard checkpoints and session files are stored
UPLOAD_DIR = "uploads"
SESSIONS_DIR = os.path.join(UPLOAD_DIR, "sessions")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Active worker tracking (heartbeats)
WORKERS = {}  # worker_id -> last_seen_epoch
HEARTBEAT_TTL = 20  # seconds considered "active"
LEASE_TTL = 120  # seconds before reclaiming an assigned shard if not completed
MAX_SHARDS_AUTO = 8  # cap for shards per dataset when auto-sizing

# Optional token-based auth (header: X-Auth-Token)
AUTH_TOKEN = os.environ.get('FINAI_DISTRIB_TOKEN', '').strip()


def ensure_csv(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()


def load_csv(file_path):
    ensure_csv(file_path)
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv(file_path, rows):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def safe_slug(name: str) -> str:
    return name.replace('/', '_').replace('\\', '_').replace(':', '_')


def dataset_seed(name: str) -> int:
    # Stable 32-bit seed from dataset name
    h = hashlib.sha256(name.encode('utf-8')).hexdigest()
    return int(h[:8], 16)


def save_session(name: str, session: dict):
    path = os.path.join(SESSIONS_DIR, f"{safe_slug(name)}.json")
    serializable = session.copy()
    # Convert sets to lists for JSON
    serializable['assigned'] = list(serializable.get('assigned', []))
    serializable['completed'] = list(serializable.get('completed', []))
    # Save leases as dict of str->float
    if 'leases' in serializable and isinstance(serializable['leases'], dict):
        serializable['leases'] = {str(k): float(v) for k, v in serializable['leases'].items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_session(name: str) -> dict | None:
    path = os.path.join(SESSIONS_DIR, f"{safe_slug(name)}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            sess = json.load(f)
        # Convert lists back to sets
        sess['assigned'] = set(sess.get('assigned', []))
        sess['completed'] = set(sess.get('completed', []))
        leases = sess.get('leases', {})
        # Convert keys back to int
        sess['leases'] = {int(k): float(v) for k, v in leases.items()} if isinstance(leases, dict) else {}
        return sess
    except Exception:
        return None


def count_active_workers() -> int:
    now = time.time()
    # prune stale
    stale = [wid for wid, ts in WORKERS.items() if (now - ts) > HEARTBEAT_TTL]
    for wid in stale:
        WORKERS.pop(wid, None)
    return max(0, len(WORKERS))


def aggregate_and_finalize(name: str, session: dict):
    """Weighted FedAvg of shard checkpoints, save aggregated model, update CSVs and session state."""
    upload_dir = session['upload_dir']
    shard_indices = sorted(session['completed'])
    if not shard_indices:
        return

    # Load all shard checkpoints and weights (num_samples)
    ckpts = []
    weights = []
    for idx in shard_indices:
        shard_path = os.path.join(upload_dir, f"shard_{idx}.pt")
        if not os.path.exists(shard_path):
            continue
        ckpt = torch.load(shard_path, map_location='cpu')
        ckpts.append(ckpt)
        shard_info = session.get('shard_info', {}).get(idx, {})
        weights.append(float(shard_info.get('num_samples', 1.0)))

    if not ckpts:
        return

    # Normalize weights
    total_w = sum(weights) if sum(weights) > 0 else float(len(ckpts))
    norm_w = [w / total_w for w in weights]

    base = ckpts[0]
    avg_state = {}
    keys = list(base['model_state_dict'].keys())
    for k in keys:
        # Initialize with zeros of the same shape
        avg_state[k] = torch.zeros_like(base['model_state_dict'][k], dtype=torch.float32)

    # Weighted sum
    for ckpt, w in zip(ckpts, norm_w):
        state = ckpt['model_state_dict']
        for k in keys:
            avg_state[k] += state[k].detach().float() * w

    # Cast back to original dtypes (use dtype from base)
    for k in keys:
        avg_state[k] = avg_state[k].to(base['model_state_dict'][k].dtype)

    # Build aggregated checkpoint reusing base metadata
    agg_ckpt = {
        'model_state_dict': avg_state,
        'vocab_size': base['vocab_size'],
        'block_size': base['block_size'],
        'is_trained': True,
    }

    out_dir = os.path.join('models', 'distributed', safe_slug(name))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'finai_gpt_fedavg.pt')
    torch.save(agg_ckpt, out_path)
    print(f"✓ Aggregated model saved to {out_path}")

    # Update CSVs: mark as trained and remove from pending datasets
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    trained = load_csv(TRAINED_CSV)
    datasets = load_csv(DATASETS_CSV)

    record = {
        'name': name,
        'config': session.get('config', ''),
        'split': session.get('split', 'train'),
        'date_trained': now,
        'model_path': out_path,
        'status': 'completed',
    }
    trained.append(record)
    save_csv(TRAINED_CSV, trained)

    remaining = [d for d in datasets if d.get('name') != name]
    save_csv(DATASETS_CSV, remaining)

    # Clear session
    if name in SESSIONS:
        del SESSIONS[name]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code=200, data=None):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        if data is not None:
            self.wfile.write(json.dumps(data).encode('utf-8'))

    def _check_auth(self) -> bool:
        if not AUTH_TOKEN:
            return True
        token = self.headers.get('X-Auth-Token', '').strip()
        return token == AUTH_TOKEN

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/next_job':
                if not self._check_auth():
                    self._send(403, {"error": "forbidden"})
                    return
                with LOCK:
                    # record heartbeat
                    qs = parse_qs(parsed.query)
                    worker_id = (qs.get('worker_id', [''])[0] or '').strip()
                    if worker_id:
                        WORKERS[worker_id] = time.time()

                    datasets = load_csv(DATASETS_CSV)
                    trained = load_csv(TRAINED_CSV)
                    trained_names = {row['name'] for row in trained}
                    pending = [d for d in datasets if d['name'] and d['name'] not in trained_names]

                    selected_job = None
                    shard_index = None
                    for d in pending:
                        name = d['name']
                        # load persisted session if any
                        if name not in SESSIONS:
                            persisted = load_session(name)
                            if persisted:
                                SESSIONS[name] = persisted
                            else:
                                # auto-size shards by active workers at assignment time
                                auto_shards = max(1, min(count_active_workers() or 1, MAX_SHARDS_AUTO))
                                SESSIONS[name] = {
                                'total_shards': auto_shards,
                                'assigned': set(),
                                'completed': set(),
                                'seed': dataset_seed(name),
                                'split': d.get('split') or 'train',
                                'config': d.get('config') or '',
                                'upload_dir': os.path.join(UPLOAD_DIR, safe_slug(name)),
                                'shard_info': {},
                                'leases': {},  # shard_index -> last lease timestamp
                            }
                            os.makedirs(SESSIONS[name]['upload_dir'], exist_ok=True)
                        sess = SESSIONS[name]
                        # Optionally expand shards if more workers joined and we still have headroom
                        active = count_active_workers()
                        if active > sess['total_shards']:
                            sess['total_shards'] = min(active, MAX_SHARDS_AUTO)
                            save_session(name, sess)
                        # Reclaim stale leases
                        now_ts = time.time()
                        to_reclaim = []
                        for idx in list(sess['assigned']):
                            if idx in sess['completed']:
                                continue
                            last = sess.get('leases', {}).get(idx, 0)
                            if last and (now_ts - last) > LEASE_TTL:
                                to_reclaim.append(idx)
                        for idx in to_reclaim:
                            sess['assigned'].discard(idx)
                            if 'leases' in sess and idx in sess['leases']:
                                del sess['leases'][idx]
                        if to_reclaim:
                            save_session(name, sess)
                        for idx in range(sess['total_shards']):
                            if idx not in sess['assigned']:
                                selected_job = d
                                shard_index = idx
                                sess['assigned'].add(idx)
                                sess.setdefault('leases', {})[idx] = time.time()
                                save_session(name, sess)
                                break
                        if selected_job is not None:
                            break

                    if selected_job is not None:
                        job = selected_job.copy()
                        job['shard_index'] = shard_index
                        job['total_shards'] = SESSIONS[selected_job['name']]['total_shards']
                        job['seed'] = SESSIONS[selected_job['name']]['seed']
                        self._send(200, {"job": job})
                    else:
                        self._send(200, {"job": None})
                return
            elif parsed.path == '/health':
                # include active worker count for visibility
                self._send(200, {"ok": True, "active_workers": count_active_workers()})
                return
            self._send(404, {"error": "not_found"})
        except Exception as e:
            print("[SERVER][GET] Unhandled error:", str(e))
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            body = self.rfile.read(length) if length else b''
            try:
                payload = json.loads(body.decode('utf-8')) if body else {}
            except Exception:
                payload = {}

            if parsed.path == '/report_shard':
                if not self._check_auth():
                    self._send(403, {"error": "forbidden"})
                    return
                qs = parse_qs(parsed.query)
                name = (qs.get('name', [''])[0] or '').strip()
                shard_index = int(qs.get('shard_index', ['0'])[0])
                total_shards = int(qs.get('total_shards', ['1'])[0])
                num_samples = float(qs.get('num_samples', ['1'])[0])
                if not name:
                    self._send(400, {"error": "missing name"})
                    return
                with LOCK:
                    sess = SESSIONS.get(name)
                    if not sess:
                        # Create session if missing (late report)
                        SESSIONS[name] = sess = {
                            'total_shards': total_shards,
                            'assigned': set(),
                            'completed': set(),
                            'seed': dataset_seed(name),
                            'split': 'train',
                            'config': '',
                            'upload_dir': os.path.join(UPLOAD_DIR, safe_slug(name)),
                            'shard_info': {},
                        }
                        os.makedirs(sess['upload_dir'], exist_ok=True)
                    # Save uploaded checkpoint bytes
                    shard_path = os.path.join(sess['upload_dir'], f"shard_{shard_index}.pt")
                    with open(shard_path, 'wb') as f:
                        f.write(body)
                    sess['completed'].add(shard_index)
                    sess['shard_info'][shard_index] = {'num_samples': num_samples}
                    # clear lease on completion
                    if 'leases' in sess and shard_index in sess['leases']:
                        del sess['leases'][shard_index]
                    save_session(name, sess)

                    if len(sess['completed']) >= sess['total_shards']:
                        # All shards done -> aggregate
                        aggregate_and_finalize(name, sess)

                self._send(200, {"ok": True})
                return

            if parsed.path == '/report_result':
                if not self._check_auth():
                    self._send(403, {"error": "forbidden"})
                    return
                name = (payload.get('name') or '').strip()
                success = bool(payload.get('success'))
                model_path = payload.get('model_path') or ''
                error_msg = payload.get('error') or ''
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if not name:
                    self._send(400, {"error": "missing name"})
                    return

                with LOCK:
                    if success:
                        trained = load_csv(TRAINED_CSV)
                        datasets = load_csv(DATASETS_CSV)
                        record = {
                            'name': name,
                            'config': payload.get('config', ''),
                            'split': payload.get('split', 'train'),
                            'date_trained': now,
                            'model_path': model_path,
                            'status': 'completed',
                        }
                        trained.append(record)
                        save_csv(TRAINED_CSV, trained)

                        remaining = [d for d in datasets if d.get('name') != name]
                        save_csv(DATASETS_CSV, remaining)
                    else:
                        # On failure: DO NOT move to trained_datasets.csv. Keep dataset in queue.
                        # Log detailed error for visibility.
                        print(f"[SERVER] Training failed for dataset '{name}': {error_msg}")

                # Respond with success flag mirroring the worker result
                self._send(200, {"ok": success, "error": error_msg})
                return

            self._send(404, {"error": "not_found"})
        except Exception as e:
            print("[SERVER][POST] Unhandled error:", str(e))
            traceback.print_exc()
            self._send(500, {"error": str(e)})


def run(host='0.0.0.0', port=8000):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Distributed training server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--max-shards', type=int, default=8, help='Upper bound for auto shards per dataset')
    parser.add_argument('--token', default=os.environ.get('FINAI_DISTRIB_TOKEN', ''), help='Auth token for workers (also read from FINAI_DISTRIB_TOKEN)')
    args = parser.parse_args()
    MAX_SHARDS_AUTO = max(1, int(args.max_shards))
    AUTH_TOKEN = (args.token or '').strip()
    run(args.host, args.port)
