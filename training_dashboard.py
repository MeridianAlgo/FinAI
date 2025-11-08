#!/usr/bin/env python3
"""
FinAI Training Dashboard
Real-time monitoring for local and distributed training
Uses only Python stdlib - no external packages required!
"""
import json
import time
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Configuration
DASHBOARD_PORT = 8080
UPDATE_INTERVAL = 10  # seconds (updates every 10 minutes as requested)
METRICS_FILE = Path("training_metrics.json")

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds is None:
        return "Calculating..."
    if seconds < 0:
        return "Complete"
    
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)

def get_training_metrics():
    """Load training metrics from file"""
    if not METRICS_FILE.exists():
        return {
            'status': 'idle',
            'message': 'No active training session'
        }
    
    try:
        with open(METRICS_FILE, 'r') as f:
            data = json.load(f)
        
        # Check if stale
        last_update = data.get('last_update', 0)
        if time.time() - last_update > 60:
            data['status'] = 'stale'
            data['message'] = 'Training metrics are stale (no update in 60s)'
        
        return data
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error loading metrics: {str(e)}'
        }

def generate_html(metrics):
    """Generate dashboard HTML"""
    status = metrics.get('status', 'idle')
    
    # Status colors
    status_colors = {
        'idle': '#6c757d',
        'training': '#28a745',
        'completed': '#007bff',
        'failed': '#dc3545',
        'stale': '#ffc107',
        'error': '#dc3545'
    }
    
    status_color = status_colors.get(status, '#6c757d')
    
    # Calculate progress percentage
    current_step = metrics.get('current_step', 0)
    total_steps = metrics.get('total_steps', 1)
    progress = (current_step / total_steps * 100) if total_steps > 0 else 0
    
    # Format times
    elapsed = format_duration(metrics.get('elapsed_time', 0))
    eta = format_duration(metrics.get('eta_seconds'))
    
    # Get loss history for chart
    loss_history = metrics.get('loss_history', [])
    loss_data = json.dumps([{'x': i, 'y': l['loss']} for i, l in enumerate(loss_history[-50:])])
    
    # Calculate average loss
    if loss_history:
        recent_losses = [l['loss'] for l in loss_history[-10:]]
        avg_loss = sum(recent_losses) / len(recent_losses)
    else:
        avg_loss = 0.0
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>FinAI Training Dashboard</title>
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
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
        
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 14px;
            background: {status_color};
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            color: #333;
            font-size: 18px;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metric {{
            margin-bottom: 15px;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            color: #333;
            font-size: 24px;
            font-weight: 600;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 14px;
        }}
        
        .chart-container {{
            width: 100%;
            height: 300px;
            position: relative;
            margin-top: 20px;
        }}
        
        .chart {{
            width: 100%;
            height: 100%;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        
        .info-item {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .info-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        
        .info-value {{
            color: #333;
            font-size: 16px;
            font-weight: 600;
        }}
        
        .timestamp {{
            color: #999;
            font-size: 12px;
            margin-top: 20px;
            text-align: center;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .training-active {{
            animation: pulse 2s ease-in-out infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FinAI Training Dashboard</h1>
            <span class="status-badge {('training-active' if status == 'training' else '')}">{status}</span>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Training Progress</h2>
                <div class="metric">
                    <div class="metric-label">Current Step</div>
                    <div class="metric-value">{current_step:,} / {total_steps:,}</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%">
                        {progress:.1f}%
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Time Tracking</h2>
                <div class="metric">
                    <div class="metric-label">Elapsed Time</div>
                    <div class="metric-value">{elapsed}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Estimated Time Remaining</div>
                    <div class="metric-value">{eta}</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Loss Metrics</h2>
                <div class="metric">
                    <div class="metric-label">Current Loss</div>
                    <div class="metric-value">{metrics.get('current_loss', 0):.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Average Loss (last 10 steps)</div>
                    <div class="metric-value">{avg_loss:.4f}</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Configuration</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Dataset</div>
                        <div class="info-value">{metrics.get('dataset_name', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Mode</div>
                        <div class="info-value">{metrics.get('training_mode', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Device</div>
                        <div class="info-value">{metrics.get('device', 'N/A')}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Learning Rate</div>
                        <div class="info-value">{metrics.get('learning_rate', 0):.2e}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Batch Size</div>
                        <div class="info-value">{metrics.get('batch_size', 0)}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Block Size</div>
                        <div class="info-value">{metrics.get('block_size', 0)}</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Loss History (Last 50 Steps)</h2>
            <div class="chart-container">
                <canvas id="lossChart" class="chart"></canvas>
            </div>
        </div>
        
        <div class="timestamp">
            Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh every {UPDATE_INTERVAL} seconds
        </div>
    </div>
    
    <script>
        // Simple canvas-based chart (no external libraries!)
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        const data = {loss_data};
        
        function drawChart() {{
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            
            if (data.length === 0) {{
                ctx.fillStyle = '#999';
                ctx.font = '16px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No data available', canvas.width / 2, canvas.height / 2);
                return;
            }}
            
            const padding = 40;
            const chartWidth = canvas.width - padding * 2;
            const chartHeight = canvas.height - padding * 2;
            
            // Find min/max
            const losses = data.map(d => d.y);
            const minLoss = Math.min(...losses);
            const maxLoss = Math.max(...losses);
            const lossRange = maxLoss - minLoss || 1;
            
            // Draw axes
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, canvas.height - padding);
            ctx.lineTo(canvas.width - padding, canvas.height - padding);
            ctx.stroke();
            
            // Draw grid
            ctx.strokeStyle = '#f0f0f0';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {{
                const y = padding + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(canvas.width - padding, y);
                ctx.stroke();
            }}
            
            // Draw line
            ctx.strokeStyle = '#667eea';
            ctx.lineWidth = 3;
            ctx.beginPath();
            
            data.forEach((point, i) => {{
                const x = padding + (i / (data.length - 1 || 1)) * chartWidth;
                const y = canvas.height - padding - ((point.y - minLoss) / lossRange) * chartHeight;
                
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            
            ctx.stroke();
            
            // Draw points
            ctx.fillStyle = '#764ba2';
            data.forEach((point, i) => {{
                const x = padding + (i / (data.length - 1 || 1)) * chartWidth;
                const y = canvas.height - padding - ((point.y - minLoss) / lossRange) * chartHeight;
                
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, Math.PI * 2);
                ctx.fill();
            }});
            
            // Draw labels
            ctx.fillStyle = '#666';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'right';
            
            // Y-axis labels
            for (let i = 0; i <= 5; i++) {{
                const value = maxLoss - (lossRange / 5) * i;
                const y = padding + (chartHeight / 5) * i;
                ctx.fillText(value.toFixed(4), padding - 10, y + 4);
            }}
            
            // X-axis label
            ctx.textAlign = 'center';
            ctx.fillText('Steps', canvas.width / 2, canvas.height - 10);
        }}
        
        drawChart();
        window.addEventListener('resize', drawChart);
    </script>
</body>
</html>"""
    
    return html

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
            metrics = get_training_metrics()
            html = generate_html(metrics)
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(html.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
            
        elif path == '/api/metrics':
            # API endpoint for metrics
            metrics = get_training_metrics()
            data = json.dumps(metrics).encode('utf-8')
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            
        else:
            self.send_response(404)
            self.end_headers()

def start_dashboard(port=DASHBOARD_PORT, open_browser=True):
    """Start the dashboard server"""
    server = HTTPServer(('localhost', port), DashboardHandler)
    
    print(f"\n{'='*80}")
    print(f"FinAI Training Dashboard")
    print(f"{'='*80}")
    print(f"Dashboard running at: http://localhost:{port}")
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
    parser = argparse.ArgumentParser(description='FinAI Training Dashboard')
    parser.add_argument('--port', type=int, default=DASHBOARD_PORT, help='Dashboard port')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    start_dashboard(port=args.port, open_browser=not args.no_browser)
