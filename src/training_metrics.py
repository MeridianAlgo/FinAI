#!/usr/bin/env python3
"""
Training Metrics Tracker
Tracks training progress and metrics for dashboard display
"""
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

class TrainingMetrics:
    """Thread-safe training metrics tracker"""
    
    def __init__(self, metrics_file: str = "training_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.lock = threading.Lock()
        self._data = {
            'status': 'idle',
            'start_time': None,
            'current_step': 0,
            'total_steps': 0,
            'current_loss': 0.0,
            'learning_rate': 0.0,
            'elapsed_time': 0,
            'eta_seconds': None,
            'dataset_name': None,
            'training_mode': None,  # 'single', 'sequential', 'all'
            'batch_size': 0,
            'block_size': 0,
            'device': 'cpu',
            'last_update': time.time(),
            'loss_history': [],  # Last 100 losses
            'step_times': [],  # Last 100 step times for ETA
            'ema_step_time': None,
        }
        self._load()
    
    def _load(self):
        """Load existing metrics if available"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    loaded = json.load(f)
                    # Only load if training is active
                    if loaded.get('status') == 'training':
                        self._data.update(loaded)
            except:
                pass
    
    def _save(self):
        """Save metrics to file"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self._data, f, indent=2)
        except:
            pass
    
    def start_training(self, dataset_name: str, total_steps: int, training_mode: str,
                      batch_size: int, block_size: int, device: str):
        """Mark training start"""
        with self.lock:
            self._data.update({
                'status': 'training',
                'start_time': time.time(),
                'current_step': 0,
                'total_steps': total_steps,
                'dataset_name': dataset_name,
                'training_mode': training_mode,
                'batch_size': batch_size,
                'block_size': block_size,
                'device': device,
                'loss_history': [],
                'step_times': [],
                'ema_step_time': None,
                'last_update': time.time(),
            })
            self._save()
    
    def update_step(self, step: int, loss: float, learning_rate: float, 
                   step_time: float, eta_seconds: Optional[float] = None):
        """Update training step metrics"""
        with self.lock:
            self._data['current_step'] = step
            self._data['current_loss'] = loss
            self._data['learning_rate'] = learning_rate
            self._data['last_update'] = time.time()
            
            # Update elapsed time
            if self._data['start_time']:
                self._data['elapsed_time'] = time.time() - self._data['start_time']
            
            # Update ETA
            if eta_seconds is not None:
                self._data['eta_seconds'] = eta_seconds
            
            # Update loss history (keep last 100)
            self._data['loss_history'].append({
                'step': step,
                'loss': loss,
                'timestamp': time.time()
            })
            if len(self._data['loss_history']) > 100:
                self._data['loss_history'] = self._data['loss_history'][-100:]
            
            # Update step times for ETA calculation
            self._data['step_times'].append(step_time)
            if len(self._data['step_times']) > 100:
                self._data['step_times'] = self._data['step_times'][-100:]
            
            # Update EMA step time
            if self._data['ema_step_time'] is None:
                self._data['ema_step_time'] = step_time
            else:
                self._data['ema_step_time'] = 0.3 * step_time + 0.7 * self._data['ema_step_time']
            
            self._save()
    
    def end_training(self, success: bool = True):
        """Mark training end"""
        with self.lock:
            self._data['status'] = 'completed' if success else 'failed'
            self._data['last_update'] = time.time()
            self._save()
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        with self.lock:
            return self._data.copy()
    
    def is_stale(self, timeout: int = 30) -> bool:
        """Check if metrics are stale (no update in timeout seconds)"""
        with self.lock:
            return (time.time() - self._data['last_update']) > timeout

# Global instance
_global_metrics = None

def get_metrics_tracker() -> TrainingMetrics:
    """Get global metrics tracker instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = TrainingMetrics()
    return _global_metrics
