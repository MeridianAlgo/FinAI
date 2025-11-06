# Distributed Training Quick Start

Get up and running with distributed training in 5 minutes.

## Prerequisites

- Raspberry Pi (or any always-on machine) for server
- 2+ machines with Python 3.8+ for workers
- All machines on same network
- FinAI installed on all machines

## Step 1: Start Server (Raspberry Pi)

```bash
cd FinAI/distributed
python server.py
```

Note the server IP address shown (e.g., `http://192.168.1.100:8765`)

## Step 2: Start Workers

On your laptop:
```bash
cd FinAI
python distributed/worker.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN
```

On your PC:
```bash
cd FinAI
python distributed/worker.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN
```

On friend's PC:
```bash
cd FinAI
python distributed/worker.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN --worker-id friend_pc
```

## Step 3: Submit Tasks

From any machine:
```bash
python distributed/client.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN submit
```

## Step 4: Monitor

**Option 1: Web Dashboard (Recommended)**
```bash
pip install flask flask-socketio
python distributed/dashboard.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN
```
Then open http://localhost:5000 in your browser

**Option 2: Command Line**
```bash
python distributed/client.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN status
```

View workers:
```bash
python distributed/client.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN workers
```

View tasks:
```bash
python distributed/client.py --server http://192.168.1.100:8765 tasks
```

## That's It!

Workers will automatically:
- Pick up tasks from the queue
- Train on assigned datasets
- Report completion to server
- Request next task

Training will be 2-3x faster with 3 workers!

## Troubleshooting

### Can't connect to server?

Check firewall on server:
```bash
# Linux/Raspberry Pi
sudo ufw allow 8765

# Windows
# Add inbound rule for port 8765 in Windows Firewall
```

### Workers not getting tasks?

Make sure you submitted tasks:
```bash
python distributed/client.py --server http://SERVER_IP:8765 submit
```

### Want to stop?

Press `Ctrl+C` on any worker or server to stop gracefully.

## Next Steps

- Read `distributed/README.md` for full documentation
- Read `distributed/EFFICIENCY_ANALYSIS.md` to understand performance
- Customize training config in `src/config.py`
