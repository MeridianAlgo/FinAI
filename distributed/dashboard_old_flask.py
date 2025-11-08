#!/usr/bin/env python3
"""
FinAI Distributed Training Web Dashboard
Real-time monitoring of distributed training system
"""
import os
import sys
import json
import time
import requests
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'finai-distributed-dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global configuration
SERVER_URL = "http://localhost:8765"
AUTH_PASSWORD = "MeridianAlgo@TRAIN"
UPDATE_INTERVAL = 2  # seconds

def get_server_status():
    """Get server status"""
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_workers():
    """Get all workers"""
    try:
        response = requests.get(
            f"{SERVER_URL}/workers",
            params={'auth_password': AUTH_PASSWORD},
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get('workers', {})
    except:
        pass
    return {}

def get_tasks():
    """Get all tasks"""
    try:
        response = requests.get(
            f"{SERVER_URL}/tasks",
            params={'auth_password': AUTH_PASSWORD},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {'pending': [], 'completed': []}

def calculate_eta(workers, pending_tasks, completed_tasks):
    """Calculate estimated time to completion"""
    if not pending_tasks:
        return "Complete", 0
    
    # Calculate average task time from completed tasks
    if completed_tasks:
        total_time = 0
        count = 0
        for task in completed_tasks:
            result = task.get('result', {})
            training_time = result.get('training_time', 0)
            if training_time > 0:
                total_time += training_time
                count += 1
        
        if count > 0:
            avg_time = total_time / count
        else:
            avg_time = 7200  # Default 2 hours
    else:
        avg_time = 7200  # Default 2 hours
    
    # Count active workers
    active_workers = sum(1 for w in workers.values() if w.get('status') in ['training', 'idle'])
    
    if active_workers == 0:
        return "No workers", 0
    
    # Calculate ETA
    remaining_tasks = len(pending_tasks)
    eta_seconds = (remaining_tasks * avg_time) / active_workers
    
    return format_duration(eta_seconds), eta_seconds

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds == 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)

def format_timestamp(timestamp):
    """Format timestamp to readable format"""
    if not timestamp:
        return "Never"
    
    dt = datetime.fromtimestamp(timestamp)
    now = datetime.now()
    diff = now - dt
    
    if diff.total_seconds() < 60:
        return f"{int(diff.total_seconds())}s ago"
    elif diff.total_seconds() < 3600:
        return f"{int(diff.total_seconds() / 60)}m ago"
    elif diff.total_seconds() < 86400:
        return f"{int(diff.total_seconds() / 3600)}h ago"
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_worker_eta(worker):
    """Get ETA for current worker task"""
    current_task = worker.get('current_task')
    if not current_task:
        return "Idle", 0
    
    assigned_at = current_task.get('assigned_at', time.time())
    elapsed = time.time() - assigned_at
    
    # Estimate based on average or default 2 hours
    estimated_total = 7200  # 2 hours default
    remaining = max(0, estimated_total - elapsed)
    
    return format_duration(remaining), remaining

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """API endpoint for status"""
    status = get_server_status()
    workers = get_workers()
    tasks = get_tasks()
    
    pending = tasks.get('pending', [])
    completed = tasks.get('completed', [])
    
    eta_str, eta_seconds = calculate_eta(workers, pending, completed)
    
    # Prepare worker data
    worker_data = []
    for worker_id, worker in workers.items():
        task_eta_str, task_eta_seconds = get_worker_eta(worker)
        
        worker_data.append({
            'id': worker_id,
            'status': worker.get('status', 'unknown'),
            'last_seen': format_timestamp(worker.get('last_seen')),
            'last_seen_raw': worker.get('last_seen', 0),
            'current_task': worker.get('current_task', {}).get('dataset', 'None'),
            'task_eta': task_eta_str,
            'task_eta_seconds': task_eta_seconds,
            'tasks_completed': worker.get('stats', {}).get('tasks_completed', 0),
            'total_time': format_duration(worker.get('stats', {}).get('total_training_time', 0)),
            'errors': worker.get('stats', {}).get('errors', 0),
            'capabilities': worker.get('capabilities', {})
        })
    
    # Sort workers by status (training first, then idle, then others)
    status_order = {'training': 0, 'idle': 1}
    worker_data.sort(key=lambda w: (status_order.get(w['status'], 2), w['id']))
    
    # Prepare task data
    pending_data = []
    for task in pending:
        pending_data.append({
            'id': task.get('task_id', 'unknown'),
            'dataset': task.get('dataset', 'unknown'),
            'submitted': format_timestamp(task.get('submitted_at'))
        })
    
    completed_data = []
    for task in completed[-20:]:  # Last 20 completed
        result = task.get('result', {})
        completed_data.append({
            'id': task.get('task_id', 'unknown'),
            'dataset': result.get('dataset', 'unknown'),
            'worker': task.get('worker_id', 'unknown'),
            'status': result.get('status', 'unknown'),
            'time': format_duration(result.get('training_time', 0)),
            'completed': format_timestamp(task.get('completed_at'))
        })
    
    completed_data.reverse()  # Most recent first
    
    return jsonify({
        'server': {
            'status': status.get('status', 'offline') if status else 'offline',
            'timestamp': format_timestamp(status.get('timestamp')) if status else 'Unknown'
        },
        'stats': {
            'workers': len(workers),
            'active_workers': sum(1 for w in workers.values() if w.get('status') in ['training', 'idle']),
            'pending_tasks': len(pending),
            'completed_tasks': len(completed),
            'eta': eta_str,
            'eta_seconds': eta_seconds
        },
        'workers': worker_data,
        'pending_tasks': pending_data,
        'completed_tasks': completed_data
    })

@app.route('/api/datasets')
def api_datasets():
    """API endpoint for available datasets"""
    try:
        response = requests.get(f"{SERVER_URL}/datasets", timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
    except:
        pass
    return jsonify({'datasets': []})

@app.route('/api/assign', methods=['POST'])
def api_assign():
    """API endpoint to assign task to worker"""
    data = request.get_json()
    worker_id = data.get('worker_id')
    dataset = data.get('dataset')
    
    try:
        response = requests.post(
            f"{SERVER_URL}/assign_task",
            json={
                'worker_id': worker_id,
                'dataset': dataset,
                'config': data.get('config', {}),
                'auth_password': AUTH_PASSWORD
            },
            timeout=10
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected")
    emit('status', {'message': 'Connected to dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected")

def background_update():
    """Background task to push updates to clients"""
    while True:
        socketio.sleep(UPDATE_INTERVAL)
        try:
            with app.app_context():
                status_data = api_status().get_json()
                socketio.emit('update', status_data, broadcast=True)
        except Exception as e:
            print(f"Error in background update: {e}")

def create_html_template():
    """Create the HTML template for the dashboard"""
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FinAI Distributed Training Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .eta-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .eta-card .stat-value {
            color: white;
        }
        
        .eta-card .stat-label {
            color: rgba(255,255,255,0.9);
        }
        
        .section {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .section-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .worker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 15px;
        }
        
        .worker-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border-left: 4px solid #667eea;
        }
        
        .worker-card.training {
            border-left-color: #28a745;
            background: #e8f5e9;
        }
        
        .worker-card.idle {
            border-left-color: #ffc107;
            background: #fff8e1;
        }
        
        .worker-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .worker-id {
            font-weight: bold;
            color: #333;
            font-size: 0.95em;
        }
        
        .worker-status {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .status-training {
            background: #28a745;
            color: white;
        }
        
        .status-idle {
            background: #ffc107;
            color: #333;
        }
        
        .worker-info {
            font-size: 0.9em;
            color: #666;
            line-height: 1.6;
        }
        
        .worker-info strong {
            color: #333;
        }
        
        .task-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        .task-table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        .task-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .task-table tr:hover {
            background: #f5f5f5;
        }
        
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .no-data {
            text-align: center;
            color: #999;
            padding: 20px;
            font-style: italic;
        }
        
        .update-time {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .connected {
            background: #28a745;
            color: white;
        }
        
        .disconnected {
            background: #dc3545;
            color: white;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .updating {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="container">
        <h1>🚀 FinAI Distributed Training Dashboard</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Server Status</div>
                <div class="stat-value" id="serverStatus">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Workers</div>
                <div class="stat-value" id="activeWorkers">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pending Tasks</div>
                <div class="stat-value" id="pendingTasks">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Completed Tasks</div>
                <div class="stat-value" id="completedTasks">0</div>
            </div>
            <div class="stat-card eta-card">
                <div class="stat-label">⏱️ ETA to Completion</div>
                <div class="stat-value" id="eta">-</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">👥 Workers (<span id="workerCount">0</span>)</div>
            <div class="worker-grid" id="workerGrid">
                <div class="no-data">No workers connected</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📋 Pending Tasks (<span id="pendingCount">0</span>)</div>
            <div id="pendingTasksTable">
                <div class="no-data">No pending tasks</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">✅ Recently Completed Tasks</div>
            <div id="completedTasksTable">
                <div class="no-data">No completed tasks</div>
            </div>
        </div>
        
        <div class="update-time">Last updated: <span id="updateTime">Never</span></div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('connect', function() {
            document.getElementById('connectionStatus').textContent = '🟢 Connected';
            document.getElementById('connectionStatus').className = 'connection-status connected';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('connectionStatus').textContent = '🔴 Disconnected';
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
        });
        
        socket.on('update', function(data) {
            updateDashboard(data);
        });
        
        function updateDashboard(data) {
            // Update stats
            document.getElementById('serverStatus').textContent = data.server.status;
            document.getElementById('activeWorkers').textContent = data.stats.active_workers;
            document.getElementById('pendingTasks').textContent = data.stats.pending_tasks;
            document.getElementById('completedTasks').textContent = data.stats.completed_tasks;
            document.getElementById('eta').textContent = data.stats.eta;
            
            // Update workers
            const workerGrid = document.getElementById('workerGrid');
            document.getElementById('workerCount').textContent = data.workers.length;
            
            if (data.workers.length === 0) {
                workerGrid.innerHTML = '<div class="no-data">No workers connected</div>';
            } else {
                workerGrid.innerHTML = data.workers.map(worker => `
                    <div class="worker-card ${worker.status}">
                        <div class="worker-header">
                            <div class="worker-id">${worker.id}</div>
                            <div class="worker-status status-${worker.status}">${worker.status}</div>
                        </div>
                        <div class="worker-info">
                            <strong>Current Task:</strong> ${worker.current_task}<br>
                            <strong>Task ETA:</strong> ${worker.task_eta}<br>
                            <strong>Last Seen:</strong> ${worker.last_seen}<br>
                            <strong>Completed:</strong> ${worker.tasks_completed} tasks<br>
                            <strong>Total Time:</strong> ${worker.total_time}<br>
                            <strong>Errors:</strong> ${worker.errors}<br>
                            ${worker.capabilities.cuda_available ? '🎮 GPU: ' + worker.capabilities.cuda_devices + ' device(s)' : '💻 CPU only'}
                        </div>
                    </div>
                `).join('');
            }
            
            // Update pending tasks
            const pendingTable = document.getElementById('pendingTasksTable');
            document.getElementById('pendingCount').textContent = data.pending_tasks.length;
            
            if (data.pending_tasks.length === 0) {
                pendingTable.innerHTML = '<div class="no-data">No pending tasks</div>';
            } else {
                pendingTable.innerHTML = `
                    <table class="task-table">
                        <thead>
                            <tr>
                                <th>Task ID</th>
                                <th>Dataset</th>
                                <th>Submitted</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.pending_tasks.map(task => `
                                <tr>
                                    <td>${task.id}</td>
                                    <td>${task.dataset}</td>
                                    <td>${task.submitted}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
            
            // Update completed tasks
            const completedTable = document.getElementById('completedTasksTable');
            
            if (data.completed_tasks.length === 0) {
                completedTable.innerHTML = '<div class="no-data">No completed tasks</div>';
            } else {
                completedTable.innerHTML = `
                    <table class="task-table">
                        <thead>
                            <tr>
                                <th>Task ID</th>
                                <th>Dataset</th>
                                <th>Worker</th>
                                <th>Status</th>
                                <th>Time</th>
                                <th>Completed</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.completed_tasks.map(task => `
                                <tr>
                                    <td>${task.id}</td>
                                    <td>${task.dataset}</td>
                                    <td>${task.worker}</td>
                                    <td><span class="status-badge status-${task.status}">${task.status}</span></td>
                                    <td>${task.time}</td>
                                    <td>${task.completed}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
            }
            
            // Update timestamp
            document.getElementById('updateTime').textContent = new Date().toLocaleTimeString();
        }
        
        // Initial load
        fetch('/api/status')
            .then(response => response.json())
            .then(data => updateDashboard(data));
    </script>
</body>
</html>'''
    
    with open(template_dir / 'dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """Start the dashboard"""
    global SERVER_URL, AUTH_PASSWORD
    
    parser = argparse.ArgumentParser(description='FinAI Distributed Training Dashboard')
    parser.add_argument('--server', default='http://localhost:8765', help='Server URL')
    parser.add_argument('--password', default='MeridianAlgo@TRAIN', help='Authentication password')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    parser.add_argument('--host', default='0.0.0.0', help='Dashboard host')
    
    args = parser.parse_args()
    
    SERVER_URL = args.server.rstrip('/')
    AUTH_PASSWORD = args.password
    
    # Create HTML template
    create_html_template()
    
    print("="*80)
    print("FinAI Distributed Training Dashboard")
    print("="*80)
    print(f"Server: {SERVER_URL}")
    print(f"Dashboard: http://localhost:{args.port}")
    print(f"Dashboard (network): http://{args.host}:{args.port}")
    print()
    print("Open the dashboard in your web browser!")
    print("Press Ctrl+C to stop")
    print("="*80)
    
    # Start background update task
    socketio.start_background_task(background_update)
    
    # Run the app
    socketio.run(app, host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
