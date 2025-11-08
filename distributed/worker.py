#!/usr/bin/env python3
"""
FinAI Distributed Training Worker
Connects to server and processes training tasks
Runs on laptop, PC, or any machine with GPU/CPU
"""
import os
import sys
import json
import time
import socket
import hashlib
import requests
import threading
from pathlib import Path
from datetime import datetime
import base64

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.finai import FinAI
from src.config import Config

class DistributedWorker:
    """Worker node for distributed training"""
    
    def __init__(self, server_url, worker_id=None, auth_password=None):
        self.server_url = server_url.rstrip('/')
        self.worker_id = worker_id or f"worker_{socket.gethostname()}_{os.getpid()}"
        self.auth_password = auth_password or 'MeridianAlgo@TRAIN'
        self.running = False
        self.current_task = None
        self.stats = {
            'tasks_completed': 0,
            'total_training_time': 0,
            'errors': 0
        }
        
        # Detect capabilities
        import torch
        self.capabilities = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'hostname': socket.gethostname(),
            'platform': sys.platform
        }
        
        print(f"Worker ID: {self.worker_id}")
        print(f"Server: {self.server_url}")
        print(f"Authentication: {'Enabled' if self.auth_password else 'Disabled'}")
        print(f"Capabilities: {self.capabilities}")
    
    def register(self):
        """Register with server"""
        try:
            response = requests.post(
                f"{self.server_url}/register",
                json={
                    'worker_id': self.worker_id,
                    'stats': self.stats,
                    'capabilities': self.capabilities,
                    'auth_password': self.auth_password
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Registered with server: {self.worker_id}")
                return True
            else:
                print(f"Registration failed: {response.text}")
                return False
        
        except Exception as e:
            print(f"Error registering: {e}")
            return False
    
    def send_heartbeat(self):
        """Send periodic heartbeat to server"""
        while self.running:
            try:
                requests.post(
                    f"{self.server_url}/heartbeat",
                    json={
                        'worker_id': self.worker_id,
                        'status': 'training' if self.current_task else 'idle',
                        'stats': self.stats,
                        'auth_password': self.auth_password
                    },
                    timeout=5
                )
            except Exception as e:
                print(f"Heartbeat error: {e}")
            
            time.sleep(30)  # Heartbeat every 30 seconds
    
    def get_task(self):
        """Request a task from server"""
        try:
            response = requests.get(
                f"{self.server_url}/get_task",
                params={
                    'worker_id': self.worker_id,
                    'auth_password': self.auth_password
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('task')
            else:
                return None
        
        except Exception as e:
            print(f"Error getting task: {e}")
            return None
    
    def complete_task(self, task_id, result):
        """Report task completion to server"""
        try:
            # Upload latest checkpoint so others continue from here
            self._upload_checkpoint_safe()
            requests.post(
                f"{self.server_url}/complete_task",
                json={
                    'worker_id': self.worker_id,
                    'task_id': task_id,
                    'result': result,
                    'auth_password': self.auth_password
                },
                timeout=10
            )
            print(f"Task {task_id} reported as complete")
        
        except Exception as e:
            print(f"Error completing task: {e}")
    
    def process_task(self, task):
        """Process a training task"""
        task_id = task['task_id']
        dataset_name = task['dataset']
        config = task.get('config', {})
        
        print(f"\n{'='*80}")
        print(f"Processing task: {task_id}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        start_time = time.time()

        try:
            # Always pull the latest unified checkpoint before training
            self._download_checkpoint_safe()
            # Import here to avoid loading on startup
            from datasets import load_dataset
            
            # Load dataset
            print(f"Loading dataset: {dataset_name}...")
            dataset_config_name = config.get('config')
            dataset_split = config.get('split', 'train')
            
            if dataset_config_name:
                dataset = load_dataset(dataset_name, dataset_config_name)
            else:
                dataset = load_dataset(dataset_name)
            
            # Extract text
            print(f"Extracting text...")
            texts = self._extract_text(dataset, dataset_split)
            
            if not texts:
                raise ValueError(f"No text data found in {dataset_name}")
            
            print(f"Extracted {len(texts):,} text samples")
            
            # Save to temp file
            os.makedirs("datasets", exist_ok=True)
            temp_file = f"datasets/worker_{self.worker_id}_{task_id}.txt"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(texts))
            
            print(f"Saved to: {temp_file}")
            
            # Train
            print(f"\nStarting training...")
            finai = FinAI()
            cfg = Config()
            
            finai.train_from_file(
                temp_file,
                steps=config.get('steps', cfg.TRAIN_STEPS),
                batch_size=config.get('batch_size', cfg.BATCH_SIZE),
                learning_rate=config.get('learning_rate', cfg.LEARNING_RATE),
                use_accelerate='auto',
                grad_accum_steps=config.get('grad_accum_steps', cfg.GRADIENT_ACCUM_STEPS),
                mixed_precision='auto',
                weight_decay=config.get('weight_decay', cfg.WEIGHT_DECAY),
                warmup_steps=config.get('warmup_steps', cfg.WARMUP_STEPS),
                max_grad_norm=config.get('max_grad_norm', cfg.MAX_GRAD_NORM)
            )
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            elapsed = time.time() - start_time
            
            result = {
                'status': 'success',
                'dataset': dataset_name,
                'samples': len(texts),
                'training_time': elapsed,
                'worker_id': self.worker_id
            }
            
            self.stats['tasks_completed'] += 1
            self.stats['total_training_time'] += elapsed
            
            print(f"\nTask completed in {elapsed:.2f}s")
            return result

        except Exception as e:
            print(f"\nTask failed: {e}")
            import traceback
            traceback.print_exc()

            self.stats['errors'] += 1
            # Upload latest checkpoint (whatever progress was saved inside training loop)
            self._upload_checkpoint_safe()
            # Report drop so task is requeued
            self._report_drop_safe(task_id, reason=str(e))

            return {
                'status': 'failed',
                'error': str(e),
                'worker_id': self.worker_id
            }
    
    def _extract_text(self, dataset, split="train"):
        """Extract text from dataset"""
        texts = []
        
        try:
            if split and split in dataset:
                data = dataset[split]
            else:
                data = dataset[list(dataset.keys())[0]]
            
            text_fields = ['text', 'input', 'question', 'instruction', 'content', 'prompt', 'query', 'answer', 'response', 'output']
            
            for item in data:
                text = None
                
                for field in text_fields:
                    if field in item and item[field]:
                        text = item[field]
                        if isinstance(text, str) and len(text.strip()) > 10:
                            break
                
                if not text or not isinstance(text, str) or len(text.strip()) < 10:
                    for key, value in item.items():
                        if isinstance(value, str) and value.strip():
                            if not text or len(value) > len(text):
                                text = value
                
                if not text or len(text.strip()) < 10:
                    text = " ".join([str(v) for k, v in item.items() if isinstance(v, (str, int, float)) and str(v).strip()])
                
                if text and isinstance(text, str) and len(text.strip()) > 10:
                    texts.append(text.strip())
        
        except Exception as e:
            print(f"Error extracting text: {e}")
        
        return texts
    
    def run(self):
        """Main worker loop"""
        print(f"\n{'='*80}")
        print("FinAI Distributed Worker")
        print(f"{'='*80}\n")
        
        # Register with server
        if not self.register():
            print("Failed to register with server. Exiting.")
            return
        
        self.running = True
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        heartbeat_thread.start()
        
        print("\nWorker started. Waiting for tasks...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # Get task from server
                task = self.get_task()

                if task:
                    self.current_task = task
                    result = self.process_task(task)
                    self.complete_task(task['task_id'], result)
                    self.current_task = None
                else:
                    # No tasks available, wait
                    time.sleep(10)

        except KeyboardInterrupt:
            print("\n\nWorker stopped by user")
            # If we were in the middle of a task, report drop and upload checkpoint
            if self.current_task:
                self._upload_checkpoint_safe()
                self._report_drop_safe(self.current_task.get('task_id'), reason='KeyboardInterrupt')
        
        finally:
            self.running = False
            print(f"\nWorker statistics:")
            print(f"  Tasks completed: {self.stats['tasks_completed']}")
            print(f"  Total training time: {self.stats['total_training_time']:.2f}s")
            print(f"  Errors: {self.stats['errors']}")

    # ----- Unified checkpoint helpers -----
    def _download_checkpoint_safe(self):
        """Download latest unified checkpoint from server if present"""
        try:
            resp = requests.get(
                f"{self.server_url}/checkpoint",
                params={'auth_password': self.auth_password},
                timeout=10
            )
            if resp.status_code == 200 and resp.content:
                models_dir = Path('models')
                models_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = models_dir / 'finai_gpt.pt'
                with open(ckpt_path, 'wb') as f:
                    f.write(resp.content)
                print(f"Pulled latest checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Checkpoint download skipped: {e}")

    def _upload_checkpoint_safe(self):
        """Upload local unified checkpoint to server (base64) if exists"""
        try:
            ckpt_path = Path('models') / 'finai_gpt.pt'
            if ckpt_path.exists():
                raw = ckpt_path.read_bytes()
                payload = {
                    'filename': 'finai_gpt.pt',
                    'content_base64': base64.b64encode(raw).decode('utf-8'),
                    'auth_password': self.auth_password
                }
                r = requests.post(f"{self.server_url}/upload_checkpoint", json=payload, timeout=20)
                if r.status_code == 200:
                    print("Uploaded checkpoint to server")
        except Exception as e:
            print(f"Checkpoint upload skipped: {e}")

    def _report_drop_safe(self, task_id, reason='unknown'):
        """Notify server that this worker dropped the task so it can be requeued"""
        try:
            requests.post(
                f"{self.server_url}/report_drop",
                json={
                    'worker_id': self.worker_id,
                    'task_id': task_id,
                    'reason': reason,
                    'auth_password': self.auth_password
                },
                timeout=10
            )
            print(f"Reported drop for task {task_id} ({reason})")
        except Exception as e:
            print(f"Drop report failed: {e}")

def main():
    """Start worker"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FinAI Distributed Training Worker')
    parser.add_argument('--server', required=True, help='Server URL (e.g., http://192.168.1.100:8765)')
    parser.add_argument('--worker-id', help='Worker ID (auto-generated if not provided)')
    parser.add_argument('--password', default='MeridianAlgo@TRAIN', help='Authentication password')
    
    args = parser.parse_args()
    
    worker = DistributedWorker(args.server, args.worker_id, args.password)
    worker.run()

if __name__ == "__main__":
    main()
