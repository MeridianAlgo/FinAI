#!/usr/bin/env python3
"""
FinAI Distributed Training Dashboard
Real-time monitoring of distributed training system
NO EXTERNAL PACKAGES - Uses only Python stdlib!
"""
import json
import time
import threading
import webbrowser
import requests
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Configuration
DASHBOARD_PORT = 8081
SERVER_URL = "http://localhost:8765"
AUTH_PASSWORD = "MeridianAlgo@TRAIN"
UPDATE_INTERVAL = 10  # seconds

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

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds is None or seconds < 0:
        return "N/A"
    if seconds == 0:
        return "0s"
    
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
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
    
    try:
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
            return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return "Unknown"

def calculate_eta(workers, pending_tasks, completed_tasks):
    """Calculate estimated time to completion with improved accuracy"""
    if not pending_tasks:
        return "Complete", 0
    
    # Calculate average task time from completed tasks
    if completed_tasks:
        total_time = 0
        count = 0
        for task in completed_tasks[-20:]:  # Use last 20 for better accuracy
            result = task.get('result', {})
            training_time = result.get('training_time', 0)
            if training_time > 0:
                total_time += training_time
                count += 1
        
        if count > 0:
            avg_time = total_time / count
        else:
            avg_time = 3600  # Default 1 hour
    else:
        avg_time = 3600  # Default 1 hour
    
    # Count active workers
    active_workers = sum(1 for w in workers.values() if w.get('status') in ['training', 'idle'])
    
    if active_workers == 0:
        return "No workers", 0
    
    # Calculate ETA
    remaining_tasks = len(pending_tasks)
    eta_seconds = (remaining_tasks * avg_time) / active_workers
    
    return format_duration(eta_seconds), eta_seconds

def generate_dashboard_html(status_data):
    """Generate dashboard HTML"""
    server_status = status_data.get('server', {})
    stats = status_data.get('stats', {})
    workers = status_data.get('workers', [])
    pending = status_data.get('pending_tasks', [])
    completed = status_data.get('completed_tasks', [])
    
    # Generate worker rows
    worker_rows = ""
    for w in workers:
        status_color = {
            'training': '#28a745',
            'idle': '#17a2b8',
            'offline': '#6c757d'
        }.get(w['status'], '#ffc107')
        
        worker_rows += f"""
        <tr>
            <td><strong>{w['id']}</strong></td>
            <td><span class="badge" style="background: {status_color}">{w['status']}</span></td>
            <td>{w['current_task']}</td>
            <td>{w['task_eta']}</td>
            <td>{w['tasks_completed']}</td>
            <td>{w['total_time']}</td>
            <td>{w['last_seen']}</td>
        </tr>
        """
    
    # Generate pending task rows
    pending_rows = ""
    for t in pending[:10]:  # Show first 10
        pending_rows += f"""
        <tr>
            <td>{t['id']}</td>
            <td>{t['dataset']}</td>
            <td>{t['submitted']}</td>
        </tr>
        """
    
    # Generate completed task rows
    completed_rows = ""
    for t in completed[:10]:  # Show first 10
        status_color = '#28a745' if t['status'] == 'completed' else '#dc3545'
        completed_rows += f"""
        <tr>
            <td>{t['id']}</td>
            <td>{t['dataset']}</td>
            <td>{t['worker']}</td>
            <td><span class="badge" style="background: {status_color}">{t['status']}</span></td>
            <td>{t['time']}</td>
            <td>{t['completed']}</td>
        </tr>
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>FinAI Distributed Training Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="{UPDATE_INTERVAL}">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }}
        
        .server-status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 14px;
            background: {('#28a745' if server_status.get('status') == 'online' else '#dc3545')};
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 36px;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            color: #333;
            font-size: 20px;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            color: #666;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            color: white;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 20px;
            text-align: center;
        }}
        
        .empty-state {{
            text-align: center;
            padding: 40px;
            color: #999;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FinAI Distributed Training Dashboard</h1>
            <span class="server-status">{server_status.get('status', 'offline')}</span>
            <p style="color: #666; margin-top: 10px;">Last server update: {server_status.get('timestamp', 'Unknown')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats.get('workers', 0)}</div>
                <div class="stat-label">Total Workers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('active_workers', 0)}</div>
                <div class="stat-label">Active Workers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('pending_tasks', 0)}</div>
                <div class="stat-label">Pending Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('completed_tasks', 0)}</div>
                <div class="stat-label">Completed Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('eta', 'N/A')}</div>
                <div class="stat-label">Estimated Time</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Workers</h2>
            {('<table><thead><tr><th>Worker ID</th><th>Status</th><th>Current Task</th><th>Task ETA</th><th>Completed</th><th>Total Time</th><th>Last Seen</th></tr></thead><tbody>' + worker_rows + '</tbody></table>') if workers else '<div class="empty-state">No workers connected</div>'}
        </div>
        
        <div class="card">
            <h2>Pending Tasks ({stats.get('pending_tasks', 0)})</h2>
            {('<table><thead><tr><th>Task ID</th><th>Dataset</th><th>Submitted</th></tr></thead><tbody>' + pending_rows + '</tbody></table>') if pending else '<div class="empty-state">No pending tasks</div>'}
        </div>
        
        <div class="card">
            <h2>Recently Completed Tasks</h2>
            {('<table><thead><tr><th>Task ID</th><th>Dataset</th><th>Worker</th><th>Status</th><th>Time</th><th>Completed</th></tr></thead><tbody>' + completed_rows + '</tbody></table>') if completed else '<div class="empty-state">No completed tasks yet</div>'}
        </div>
        
        <div class="timestamp">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh every {UPDATE_INTERVAL} seconds
        </div>
    </div>
</body>
</html>"""
    
    return html

def get_dashboard_data():
    """Collect all dashboard data"""
    status = get_server_status()
    workers = get_workers()
    tasks = get_tasks()
    
    pending = tasks.get('pending', [])
    completed = tasks.get('completed', [])
    
    eta_str, eta_seconds = calculate_eta(workers, pending, completed)
    
    # Prepare worker data
    worker_data = []
    for worker_id, worker in workers.items():
        current_task = worker.get('current_task', {})
        task_dataset = current_task.get('dataset', 'None')
        
        # Calculate task ETA
        if current_task:
            assigned_at = current_task.get('assigned_at', time.time())
            elapsed = time.time() - assigned_at
            # Use average from completed tasks or default
            if completed:
                avg_time = sum(t.get('result', {}).get('training_time', 3600) for t in completed[-10:]) / min(10, len(completed))
            else:
                avg_time = 3600
            remaining = max(0, avg_time - elapsed)
            task_eta = format_duration(remaining)
        else:
            task_eta = "Idle"
        
        worker_data.append({
            'id': worker_id,
            'status': worker.get('status', 'unknown'),
            'last_seen': format_timestamp(worker.get('last_seen')),
            'current_task': task_dataset,
            'task_eta': task_eta,
            'tasks_completed': worker.get('stats', {}).get('tasks_completed', 0),
            'total_time': format_duration(worker.get('stats', {}).get('total_training_time', 0)),
        })
    
    # Sort workers by status
    status_order = {'training': 0, 'idle': 1}
    worker_data.sort(key=lambda w: (status_order.get(w['status'], 2), w['id']))
    
    # Prepare task data
    pending_data = []
    for task in pending[:20]:
        pending_data.append({
            'id': task.get('task_id', 'unknown'),
            'dataset': task.get('dataset', 'unknown'),
            'submitted': format_timestamp(task.get('submitted_at'))
        })
    
    completed_data = []
    for task in completed[-20:]:
        result = task.get('result', {})
        completed_data.append({
            'id': task.get('task_id', 'unknown'),
            'dataset': result.get('dataset', 'unknown'),
            'worker': task.get('worker_id', 'unknown'),
            'status': result.get('status', 'unknown'),
            'time': format_duration(result.get('training_time', 0)),
            'completed': format_timestamp(task.get('completed_at'))
        })
    
    completed_data.reverse()
    
    return {
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
    }

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard"""
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        if path == '/':
            # Serve dashboard HTML
            try:
                data = get_dashboard_data()
                html = generate_dashboard_html(data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(html.encode('utf-8'))))
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            except Exception as e:
                error_html = f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
                self.send_response(500)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(error_html.encode('utf-8'))
        
        elif path == '/api/status':
            # API endpoint for status
            try:
                data = get_dashboard_data()
                json_data = json.dumps(data).encode('utf-8')
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(json_data)))
                self.end_headers()
                self.wfile.write(json_data)
            except Exception as e:
                error_data = json.dumps({'error': str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(error_data)
        
        else:
            self.send_response(404)
            self.end_headers()

def start_dashboard(port=DASHBOARD_PORT, open_browser=True):
    """Start the dashboard server"""
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    
    print(f"\n{'='*80}")
    print(f"FinAI Distributed Training Dashboard")
    print(f"{'='*80}")
    print(f"Dashboard running at: http://localhost:{port}")
    print(f"Server URL: {SERVER_URL}")
    print(f"Auto-refresh interval: {UPDATE_INTERVAL} seconds")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*80}\n")
    
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down dashboard...")
        server.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='FinAI Distributed Training Dashboard')
    parser.add_argument('--port', type=int, default=DASHBOARD_PORT, help='Dashboard port')
    parser.add_argument('--server', type=str, default=SERVER_URL, help='Server URL')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    SERVER_URL = args.server
    start_dashboard(port=args.port, open_browser=not args.no_browser)
