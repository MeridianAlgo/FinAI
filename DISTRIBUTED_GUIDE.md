# FinAI Distributed Training Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Network Setup](#network-setup)
5. [Server Setup](#server-setup)
6. [Worker Setup](#worker-setup)
7. [Remote Access Setup](#remote-access-setup)
8. [Authentication & Security](#authentication--security)
9. [Task Management](#task-management)
10. [Monitoring Progress](#monitoring-progress)
11. [Troubleshooting](#troubleshooting)
12. [Advanced Configuration](#advanced-configuration)
13. [FAQs](#frequently-asked-questions)

## Introduction

This guide provides **comprehensive, step-by-step instructions** for setting up distributed training across multiple computers. The system uses a server-worker architecture where a central server coordinates training tasks across multiple worker nodes.

**Key Features:**
- **Secure Authentication**: Password-protected server (`MeridianAlgo@TRAIN`)
- **Remote Access**: Workers can connect from anywhere (different homes/networks)
- **Easy Setup**: Works on Windows, macOS, Linux, and Raspberry Pi
- **2-3x Speedup**: Train on multiple datasets simultaneously

## System Requirements

### Server (Raspberry Pi or any computer)
- **Minimum:**
  - 1GB RAM
  - 10GB free storage
  - Ethernet connection (recommended)
  - Python 3.8+
- **Recommended:**
  - Raspberry Pi 4 (4GB+ RAM)
  - 32GB+ storage
  - Ubuntu Server or Raspberry Pi OS
  - Static IP address

### Worker Computers
- **Minimum:**
  - 8GB RAM
  - 20GB free storage
  - Python 3.8+
  - Stable internet connection
- **Recommended:**
  - 16GB+ RAM
  - NVIDIA GPU (CUDA compatible)
  - 100GB+ free storage
  - Ubuntu 20.04+ or Windows 10/11

## Quick Start

### For Local Network (Same WiFi/Ethernet)

**1. Server (Your Computer):**
```bash
cd FinAI
python distributed/server.py
```
Note the IP address shown (e.g., `http://192.168.1.100:8765`)

**2. Worker (Friend's Laptop on Same Network):**
```bash
cd FinAI
python distributed/worker.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN
```

**3. Submit Tasks:**
```bash
python distributed/client.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN submit
```

### For Remote Access (Different Networks/Homes)

**1. Server (Your Computer):**
```bash
# Forward port 8765 on your router (see Remote Access Setup section)
cd FinAI
python distributed/server.py
```
Note your public IP (visit https://whatismyipaddress.com)

**2. Worker (Friend's Computer at Their Home):**
```bash
cd FinAI
python distributed/worker.py --server http://YOUR_PUBLIC_IP:8765 --password MeridianAlgo@TRAIN
```

**3. Submit Tasks:**
```bash
python distributed/client.py --server http://YOUR_PUBLIC_IP:8765 --password MeridianAlgo@TRAIN submit
```

**That's it!** Workers will automatically start training on assigned datasets.

## Network Setup

### 1. Local Network Configuration
- Ensure all computers are on the same local network
- For best performance, use wired Ethernet connections
- If using WiFi, ensure strong signal strength (at least 3 bars)

### 2. Finding IP Addresses
- **Windows:**
  ```cmd
  ipconfig
  ```
  Look for "IPv4 Address" under your active network connection

- **macOS/Linux:**
  ```bash
  ifconfig | grep "inet " | grep -v 127.0.0.1
  ```
  or
  ```bash
  hostname -I
  ```

### 3. Port Forwarding (For Remote Workers)
See the [Remote Access Setup](#remote-access-setup) section for detailed instructions.

## Server Setup

### 1. Initial Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv git

# Clone the repository
git clone https://github.com/your-username/FinAI.git
cd FinAI

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure the Server

The server configuration is in `distributed/server_config.json`:
```json
{
  "auth_enabled": true,
  "auth_password": "MeridianAlgo@TRAIN",
  "max_workers": 50,
  "worker_timeout": 300,
  "log_level": "INFO",
  "server_port": 8765,
  "allow_remote_workers": true
}
```

**Important Configuration Options:**
- `auth_enabled`: Set to `true` to require password authentication (RECOMMENDED)
- `auth_password`: The password workers must use (`MeridianAlgo@TRAIN`)
- `max_workers`: Maximum number of simultaneous workers
- `worker_timeout`: Remove workers after this many seconds of inactivity (300 = 5 minutes)
- `allow_remote_workers`: Allow workers from different networks

**To change the password:**
Edit `distributed/server_config.json` and change `auth_password` to your desired password.

### 3. Start the Server
```bash
# Navigate to project directory
cd FinAI

# Activate virtual environment
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Start the server
python distributed/server.py
```

### 4. Run as a Service (Optional, for 24/7 Operation)
Create a systemd service file at `/etc/systemd/system/finai-server.service`:
```ini
[Unit]
Description=FinAI Distributed Training Server
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/FinAI
ExecStart=/home/pi/FinAI/venv/bin/python distributed/server.py
Restart=always
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable finai-server
sudo systemctl start finai-server
```

## Worker Setup

### 1. Initial Setup (Windows/macOS/Linux)
```bash
# Clone the repository
git clone https://github.com/your-username/FinAI.git
cd FinAI

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure the Worker (Optional)

The worker configuration is in `distributed/worker_config.json`:
```json
{
  "server_url": "http://YOUR_SERVER_IP:8765",
  "auth_password": "MeridianAlgo@TRAIN",
  "worker_id": "auto",
  "heartbeat_interval": 30,
  "task_poll_interval": 10,
  "gpu_enabled": "auto"
}
```

**Configuration is optional** - you can pass everything via command line arguments.

### 3. Start the Worker

**Basic start (with password):**
```bash
python distributed/worker.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN
```

**With custom worker ID:**
```bash
python distributed/worker.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN --worker-id laptop_john
```

**For remote worker (different network):**
```bash
python distributed/worker.py --server http://YOUR_PUBLIC_IP:8765 --password MeridianAlgo@TRAIN
```

**Run in background (Linux/macOS):**
```bash
nohup python distributed/worker.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN > worker.log 2>&1 &
```

**Run in background (Windows):**
```powershell
Start-Process python -ArgumentList "distributed/worker.py","--server","http://SERVER_IP:8765","--password","MeridianAlgo@TRAIN" -WindowStyle Hidden
```

### 4. Windows-Specific Setup
For better performance on Windows:
1. Disable Windows Defender real-time protection during training
2. Run PowerShell as Administrator and execute:
   ```powershell
   Set-ExecutionPolicy RemoteSigned
   ```

## Remote Access Setup

This section explains how to allow workers from **different networks** (e.g., friends at their homes) to connect to your server.

### Prerequisites
- Your server must be accessible from the internet
- You need access to your router's admin panel
- You should have a static local IP for your server (or DHCP reservation)

### Step 1: Find Your Server's Local IP

**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" (e.g., `192.168.1.100`)

**Linux/Raspberry Pi:**
```bash
hostname -I
```

### Step 2: Configure Port Forwarding on Your Router

1. **Access your router:**
   - Open browser and go to `http://192.168.1.1` (or `192.168.0.1`, `10.0.0.1`)
   - Log in with admin credentials

2. **Find Port Forwarding section:**
   - Look for: "Port Forwarding", "Virtual Server", "NAT Forwarding", or "Applications & Gaming"

3. **Create a new rule:**
   - **Service Name**: FinAI Server
   - **External Port**: 8765
   - **Internal Port**: 8765
   - **Internal IP**: Your server's local IP (e.g., 192.168.1.100)
   - **Protocol**: TCP
   - **Enabled**: Yes

4. **Save and apply**

### Step 3: Configure Windows Firewall

**Windows:**
```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "FinAI Server" -Direction Inbound -LocalPort 8765 -Protocol TCP -Action Allow
```

**Linux/Raspberry Pi:**
```bash
sudo ufw allow 8765/tcp
sudo ufw reload
```

### Step 4: Find Your Public IP Address

Visit https://whatismyipaddress.com or run:
```bash
curl https://api.ipify.org
```

Your public IP might look like: `203.0.113.45`

### Step 5: Test Remote Access

From a device on a **different network** (e.g., mobile hotspot):
```bash
curl http://YOUR_PUBLIC_IP:8765/status
```

You should see:
```json
{"status": "online", "workers": 0, "pending_tasks": 0, "completed_tasks": 0, "auth_required": true}
```

### Step 6: Share Connection Info with Remote Workers

Send your friends:
1. **Server URL**: `http://YOUR_PUBLIC_IP:8765`
2. **Password**: `MeridianAlgo@TRAIN`
3. **Instructions**: See [Worker Setup](#worker-setup)

### Dynamic DNS (Optional)

If your public IP changes frequently, use a free Dynamic DNS service:
- **DuckDNS**: https://www.duckdns.org (recommended)
- **No-IP**: https://www.noip.com
- **Dynu**: https://www.dynu.com

With DDNS, workers can connect to `http://yourname.duckdns.org:8765` instead of an IP address.

### Security Notes for Remote Access

- **Password is required** - Workers MUST use `--password MeridianAlgo@TRAIN`
- **Monitor server logs** for unauthorized access attempts
- **Only share credentials with trusted users**
- **Consider changing the password** in `distributed/server_config.json`

## Authentication & Security

### Password Authentication

The server requires password authentication by default to prevent unauthorized access.

**Default Password:** `MeridianAlgo@TRAIN`

### How Authentication Works

1. **Server** checks password on all requests (except `/status`)
2. **Workers** must provide password when:
   - Registering with server
   - Sending heartbeats
   - Requesting tasks
   - Completing tasks

3. **Client** must provide password when:
   - Submitting tasks
   - Viewing workers
   - Viewing tasks

### Changing the Password

**1. Edit server config:**
```bash
nano distributed/server_config.json
```

Change `auth_password`:
```json
{
  "auth_password": "YOUR_NEW_STRONG_PASSWORD"
}
```

**2. Restart server:**
```bash
python distributed/server.py
```

**3. Update all workers and clients:**
```bash
# Workers
python distributed/worker.py --server http://SERVER_IP:8765 --password YOUR_NEW_STRONG_PASSWORD

# Client
python distributed/client.py --server http://SERVER_IP:8765 --password YOUR_NEW_STRONG_PASSWORD submit
```

### Disabling Authentication (NOT RECOMMENDED)

Edit `distributed/server_config.json`:
```json
{
  "auth_enabled": false
}
```

**Warning:** Only disable authentication on a trusted, isolated network.

### Security Best Practices

1. **Use a strong password** (mix of letters, numbers, symbols)
2. **Change the default password** before allowing remote access
3. **Monitor server logs** regularly
4. **Only share credentials with trusted users**
5. **Use HTTPS** for production (requires reverse proxy like nginx)
6. **Keep software updated** (`git pull && pip install -r requirements.txt --upgrade`)
7. **Disable server when not in use** (Press Ctrl+C)

### Monitoring for Unauthorized Access

Check server logs:
```bash
# Server shows authentication failures
tail -f distributed_data/logs/server.log | grep "Authentication failed"
```

Failed attempts will show:
```
[2025-11-05 19:30:45] Authentication failed from 203.0.113.99
```

## Task Management

### 1. Submitting Tasks

**Submit all pending datasets:**
```bash
python distributed/client.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN submit
```

This will:
1. Read `datasets.csv`
2. Check `trained_datasets.csv` for already-trained datasets
3. Submit all pending datasets as tasks
4. Workers will automatically pick them up

**Example output:**
```
Found 22 pending datasets

Submitted: yukiarimo/english-vocabulary (task_id: a1b2c3d4)
Submitted: vumichien/financial-sentiment (task_id: e5f6g7h8)
Submitted: TimKoornstra/financial-tweets-sentiment (task_id: i9j0k1l2)
...

Submitted 22/22 tasks
```

## Monitoring Progress

### 1. Command Line Interface

**Show server status (no password required):**
```bash
python distributed/client.py --server http://SERVER_IP:8765 status
```

Output:
```
================================================================================
Server Status
================================================================================
Status: online
Workers: 3
Pending tasks: 19
Completed tasks: 3
```

**List all workers:**
```bash
python distributed/client.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN workers
```

Output:
```
================================================================================
Workers (3)
================================================================================

worker_LAPTOP-ABC_12345:
  Status: training
  Current task: yukiarimo/english-vocabulary
  Tasks completed: 2
  Total training time: 7200.45s

worker_PC-XYZ_67890:
  Status: idle
  Current task: None
  Tasks completed: 3
  Total training time: 10800.12s
```

**List all tasks:**
```bash
python distributed/client.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN tasks
```

Output:
```
================================================================================
Tasks
================================================================================

Pending (19):
  a1b2c3d4: yukiarimo/english-vocabulary
  e5f6g7h8: vumichien/financial-sentiment
  ...

Completed (3):
  i9j0k1l2: success by worker_LAPTOP-ABC_12345
  m3n4o5p6: success by worker_PC-XYZ_67890
  ...
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused
```
ConnectionError: [Errno 111] Connection refused
```
**Solution:**
- Verify server is running: `python distributed/server.py`
- Check firewall settings (see [Remote Access Setup](#remote-access-setup))
- Ensure correct IP address and port
- Test with: `curl http://SERVER_IP:8765/status`

#### 2. Authentication Failed
```
{'error': 'Authentication failed. Invalid password.'}
```
**Solution:**
- Verify password is correct: `MeridianAlgo@TRAIN`
- Check server config: `cat distributed/server_config.json`
- Ensure you're using `--password` argument:
  ```bash
  python distributed/worker.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN
  ```

#### 3. Module Not Found
```
ModuleNotFoundError: No module named 'torch'
```
**Solution:**
```bash
pip install -r requirements.txt
```

#### 4. CUDA Errors
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size in `src/config.py`
- Close other GPU-intensive applications
- Use CPU instead (automatic fallback)

#### 5. Port Already in Use
```
OSError: [Errno 98] Address already in use
```
**Solution:**
```bash
# Find process using port 8765
# Windows:
netstat -ano | findstr :8765

# Linux/macOS:
lsof -i :8765

# Kill the process or use a different port
```

#### 6. Workers Not Getting Tasks
**Solution:**
- Check tasks were submitted: `python distributed/client.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN tasks`
- Verify workers are registered: `python distributed/client.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN workers`
- Check worker logs for errors

#### 7. Remote Workers Can't Connect
**Solution:**
- Verify port forwarding is configured correctly
- Test from external network: `curl http://YOUR_PUBLIC_IP:8765/status`
- Check firewall allows incoming connections on port 8765
- Ensure server is running and accessible
- Verify public IP hasn't changed

### Logs
- **Server logs**: Console output (or redirect to file)
- **Worker logs**: Console output (or use `nohup` to save to file)
- **Data directory**: `distributed_data/` (created automatically)

## Advanced Configuration

### 1. Custom Training Parameters
Create a `training_config.json`:
```json
{
    "batch_size": 32,
    "learning_rate": 0.0001,
    "num_epochs": 3,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "bf16"
}
```

### 2. Resource Limits
```bash
# Limit GPU memory usage (e.g., 4GB)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Limit CPU threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 3. Docker Support
```dockerfile
# Dockerfile.worker
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "distributed/worker.py", "--server", "http://host.docker.internal:8765"]
```

## Security Considerations

### 1. Network Security
- Use a VPN for remote workers
- Enable TLS/SSL for server communication
- Implement API key authentication

### 2. Data Security
- Encrypt sensitive data
- Use .env files for secrets
- Regularly update dependencies

### 3. Access Control
```ini
# distributed/config.ini
[auth]
enabled = true
api_key = your-secure-api-key
allowed_ips = 192.168.1.0/24
```

## Frequently Asked Questions

### Q: How do I add more workers?
A: Simply start the worker script on additional machines using the same server URL and password:
```bash
python distributed/worker.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN
```

### Q: Can workers be on different networks?
A: Yes! Workers can connect from anywhere if you:
1. Forward port 8765 on your router
2. Use your public IP address
3. Provide the correct password

See [Remote Access Setup](#remote-access-setup) for details.

### Q: How do I know if my workers are connected?
A: Check worker status:
```bash
python distributed/client.py --server http://SERVER_IP:8765 --password MeridianAlgo@TRAIN workers
```

### Q: Can I pause training?
A: Yes, press Ctrl+C on any worker to stop it. The task will remain in the queue and another worker will pick it up.

### Q: How do I update the code on all machines?
```bash
git pull
pip install -r requirements.txt --upgrade
```

### Q: Where are models saved?
A: Each worker saves to its local `models/finai_gpt.pt`. All workers train the same shared model (loaded from server or local copy).

### Q: What happens if a worker crashes?
A: The server will mark it as stale after 5 minutes. The task will remain in the queue for another worker to pick up.

### Q: Can I use this on a VPS/cloud server?
A: Yes! Deploy the server on a VPS (AWS, DigitalOcean, etc.) and workers can connect from anywhere.

### Q: How secure is the password authentication?
A: The system uses constant-time password comparison to prevent timing attacks. For production use, consider adding HTTPS with a reverse proxy.

### Q: Can I change the port from 8765?
A: Yes, edit `distributed/server_config.json` and change `server_port`, then update all worker/client commands.

### Q: What if my public IP changes?
A: Use a Dynamic DNS service (DuckDNS, No-IP) to get a permanent hostname. See [Remote Access Setup](#remote-access-setup).

### Q: How many workers can I have?
A: The default limit is 50 workers (configurable in `server_config.json`). The server is lightweight and can handle many workers.

### Q: Do I need a GPU?
A: No, workers will automatically use CPU if no GPU is available. Training will be slower but still works.

### Q: Can I run multiple workers on the same machine?
A: Yes, but it's usually better to run one worker per machine to avoid resource contention.

### Q: How do I stop the server?
A: Press Ctrl+C in the terminal where the server is running.

### Q: Can I see training progress on workers?
A: Yes, each worker shows detailed training progress in its console output, including ETA and loss.

## Complete Example Workflow

### Scenario: You + 2 Friends Training Together

**Your Setup (Server + Worker):**
```bash
# Terminal 1: Start server
cd FinAI
python distributed/server.py

# Note your public IP: 203.0.113.45

# Terminal 2: Start your worker
python distributed/worker.py --server http://localhost:8765 --password MeridianAlgo@TRAIN --worker-id my_laptop
```

**Friend 1 (At Their Home):**
```bash
# Clone repo
git clone https://github.com/your-username/FinAI.git
cd FinAI
pip install -r requirements.txt

# Start worker
python distributed/worker.py --server http://203.0.113.45:8765 --password MeridianAlgo@TRAIN --worker-id friend1_pc
```

**Friend 2 (At Their Home):**
```bash
# Clone repo
git clone https://github.com/your-username/FinAI.git
cd FinAI
pip install -r requirements.txt

# Start worker
python distributed/worker.py --server http://203.0.113.45:8765 --password MeridianAlgo@TRAIN --worker-id friend2_laptop
```

**Submit Tasks (You):**
```bash
python distributed/client.py --server http://localhost:8765 --password MeridianAlgo@TRAIN submit
```

**Monitor Progress (Anyone):**
```bash
python distributed/client.py --server http://203.0.113.45:8765 --password MeridianAlgo@TRAIN status
python distributed/client.py --server http://203.0.113.45:8765 --password MeridianAlgo@TRAIN workers
```

**Result:**
- 22 datasets trained in ~16 hours instead of 44 hours
- 2.75x speedup with 3 workers
- Everyone contributes to the same model

## Getting Help

For support, please open an issue at:
https://github.com/your-username/FinAI/issues

## Summary

**What You've Learned:**
1. ✅ How to set up a distributed training server
2. ✅ How to connect workers from anywhere (local or remote)
3. ✅ How to secure your server with password authentication
4. ✅ How to submit and monitor training tasks
5. ✅ How to troubleshoot common issues
6. ✅ How to configure port forwarding for remote access

**Next Steps:**
1. Start your server
2. Connect workers (local or remote)
3. Submit tasks
4. Monitor progress
5. Enjoy 2-3x faster training!

---
*Last updated: November 5, 2025*
