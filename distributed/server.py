#!/usr/bin/env python3
"""
FinAI Distributed Training Server
Coordinates training across multiple worker nodes
Can run on Raspberry Pi or any always-on machine
"""
import os
import json
import time
import socket
import threading
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Server configuration
SERVER_PORT = 8765
DATA_DIR = Path("distributed_data")
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"
LOGS_DIR = DATA_DIR / "logs"
CONFIG_FILE = Path("distributed") / "server_config.json"

# Load configuration
def load_config():
    """Load server configuration from file"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        'auth_enabled': True,
        'auth_password': 'MeridianAlgo@TRAIN',
        'max_workers': 50,
        'worker_timeout': 300,
        'log_level': 'INFO'
    }

config = load_config()

# Create directories
for d in [DATA_DIR, MODELS_DIR, DATASETS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Global state
workers = {}  # worker_id -> {last_seen, status, current_task, stats}
task_queue = []  # List of pending tasks
completed_tasks = []  # List of completed tasks
model_shards = {}  # shard_id -> {worker_id, status, checksum}

class DistributedTrainingServer(BaseHTTPRequestHandler):
    """HTTP server for coordinating distributed training"""
    
    def log_message(self, format, *args):
        """Override to add timestamps to logs"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {format % args}")
    
    def _send_json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _verify_auth(self, data):
        """Verify authentication token"""
        if not config.get('auth_enabled', True):
            return True
        
        provided_password = data.get('auth_password', '')
        expected_password = config.get('auth_password', 'MeridianAlgo@TRAIN')
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(provided_password, expected_password)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        
        # Check auth for protected endpoints
        if path != '/status' and config.get('auth_enabled', True):
            auth_password = params.get('auth_password', [''])[0]
            if not hmac.compare_digest(auth_password, config.get('auth_password', 'MeridianAlgo@TRAIN')):
                self._send_json({'error': 'Authentication failed'}, 401)
                return
        
        if path == '/status':
            # Server status (public endpoint)
            self._send_json({
                'status': 'online',
                'workers': len(workers),
                'pending_tasks': len(task_queue),
                'completed_tasks': len(completed_tasks),
                'timestamp': time.time(),
                'auth_required': config.get('auth_enabled', True)
            })
        
        elif path == '/workers':
            # List all workers
            self._send_json({
                'workers': {
                    wid: {
                        'last_seen': w['last_seen'],
                        'status': w['status'],
                        'current_task': w.get('current_task'),
                        'stats': w.get('stats', {})
                    }
                    for wid, w in workers.items()
                }
            })
        
        elif path == '/tasks':
            # List all tasks
            self._send_json({
                'pending': task_queue,
                'completed': completed_tasks
            })
        
        elif path == '/get_task':
            # Worker requesting a task
            worker_id = params.get('worker_id', ['unknown'])[0]
            
            # Update worker heartbeat
            if worker_id in workers:
                workers[worker_id]['last_seen'] = time.time()
            
            if task_queue:
                task = task_queue.pop(0)
                task['assigned_to'] = worker_id
                task['assigned_at'] = time.time()
                
                if worker_id in workers:
                    workers[worker_id]['current_task'] = task
                    workers[worker_id]['status'] = 'training'
                
                self._send_json({'task': task})
            else:
                self._send_json({'task': None, 'message': 'No tasks available'})
        
        else:
            self._send_json({'error': 'Unknown endpoint'}, 404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({'error': 'Invalid JSON'}, 400)
            return
        
        # Verify authentication for all POST endpoints
        if config.get('auth_enabled', True) and not self._verify_auth(data):
            self._send_json({'error': 'Authentication failed. Invalid password.'}, 401)
            self.log_message(f"Authentication failed from {self.client_address[0]}")
            return
        
        if path == '/register':
            # Worker registration
            worker_id = data.get('worker_id')
            if not worker_id:
                self._send_json({'error': 'worker_id required'}, 400)
                return
            
            workers[worker_id] = {
                'registered_at': time.time(),
                'last_seen': time.time(),
                'status': 'idle',
                'current_task': None,
                'stats': data.get('stats', {}),
                'capabilities': data.get('capabilities', {})
            }
            
            self.log_message(f"Worker registered: {worker_id}")
            self._send_json({'status': 'registered', 'worker_id': worker_id})
        
        elif path == '/heartbeat':
            # Worker heartbeat
            worker_id = data.get('worker_id')
            if worker_id in workers:
                workers[worker_id]['last_seen'] = time.time()
                workers[worker_id]['status'] = data.get('status', 'idle')
                self._send_json({'status': 'ok'})
            else:
                self._send_json({'error': 'Worker not registered'}, 404)
        
        elif path == '/submit_task':
            # Submit a new training task
            task = {
                'task_id': hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
                'dataset': data.get('dataset'),
                'config': data.get('config', {}),
                'submitted_at': time.time(),
                'status': 'pending'
            }
            task_queue.append(task)
            self.log_message(f"Task submitted: {task['task_id']} - {task['dataset']}")
            self._send_json({'status': 'submitted', 'task_id': task['task_id']})
        
        elif path == '/complete_task':
            # Worker completed a task
            worker_id = data.get('worker_id')
            task_id = data.get('task_id')
            result = data.get('result', {})
            
            if worker_id in workers:
                workers[worker_id]['current_task'] = None
                workers[worker_id]['status'] = 'idle'
                workers[worker_id]['last_seen'] = time.time()
            
            completed_tasks.append({
                'task_id': task_id,
                'worker_id': worker_id,
                'completed_at': time.time(),
                'result': result
            })
            
            self.log_message(f"Task completed: {task_id} by {worker_id}")
            self._send_json({'status': 'recorded'})
        
        elif path == '/upload_shard':
            # Worker uploading model shard
            shard_id = data.get('shard_id')
            worker_id = data.get('worker_id')
            checksum = data.get('checksum')
            
            model_shards[shard_id] = {
                'worker_id': worker_id,
                'uploaded_at': time.time(),
                'checksum': checksum,
                'status': 'available'
            }
            
            self.log_message(f"Shard uploaded: {shard_id} from {worker_id}")
            self._send_json({'status': 'uploaded', 'shard_id': shard_id})
        
        else:
            self._send_json({'error': 'Unknown endpoint'}, 404)

def cleanup_stale_workers():
    """Remove workers that haven't sent heartbeat in 5 minutes"""
    while True:
        time.sleep(60)  # Check every minute
        now = time.time()
        stale = []
        
        for worker_id, worker in workers.items():
            if now - worker['last_seen'] > 300:  # 5 minutes
                stale.append(worker_id)
        
        for worker_id in stale:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Removing stale worker: {worker_id}")
            del workers[worker_id]

def main():
    """Start the distributed training server"""
    print("="*80)
    print("FinAI Distributed Training Server")
    print("="*80)
    print(f"Starting server on port {SERVER_PORT}...")
    print(f"Data directory: {DATA_DIR.absolute()}")
    print(f"Authentication: {'ENABLED' if config.get('auth_enabled', True) else 'DISABLED'}")
    if config.get('auth_enabled', True):
        print(f"Password: {config.get('auth_password', 'MeridianAlgo@TRAIN')}")
    print()
    
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Local address: http://{local_ip}:{SERVER_PORT}")
    print(f"External address: http://YOUR_PUBLIC_IP:{SERVER_PORT}")
    print(f"Workers can connect from anywhere using your public IP")
    print()
    print("Endpoints:")
    print("  GET  /status        - Server status")
    print("  GET  /workers       - List workers")
    print("  GET  /tasks         - List tasks")
    print("  GET  /get_task      - Get next task (worker)")
    print("  POST /register      - Register worker")
    print("  POST /heartbeat     - Worker heartbeat")
    print("  POST /submit_task   - Submit training task")
    print("  POST /complete_task - Mark task complete")
    print("  POST /upload_shard  - Upload model shard")
    print()
    print("Press Ctrl+C to stop")
    print("="*80)
    print()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_stale_workers, daemon=True)
    cleanup_thread.start()
    
    # Start HTTP server
    server = HTTPServer(('0.0.0.0', SERVER_PORT), DistributedTrainingServer)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.shutdown()
        print("Server stopped")

if __name__ == "__main__":
    main()
