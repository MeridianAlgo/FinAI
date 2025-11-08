# FinAI Distributed Training

Train your FinAI model across multiple machines for faster training.

## Architecture

```
┌─────────────────┐
│  Server Node    │  (Raspberry Pi or always-on machine)
│  - Coordinates  │
│  - Task queue   │
│  - Sync state   │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│Worker1│ │Worker2│ │Worker3│ │Worker4│
│Laptop │ │Your PC│ │Friend │ │Cloud │
└───────┘ └──────┘ └──────┘ └──────┘
```

## How It Works

### Task Distribution (NOT Data Parallelism)
Each worker trains on a **different dataset** sequentially:
- Worker 1: Trains on `dataset_A` → saves model shard
- Worker 2: Trains on `dataset_B` → saves model shard  
- Worker 3: Trains on `dataset_C` → saves model shard
- Server: Merges shards into final model

**Benefits:**
- Faster overall training (parallel dataset processing)
- Each worker trains independently
- No complex gradient synchronization
- Works well with heterogeneous hardware

**Note:** This is NOT traditional distributed training (data parallelism). Each worker processes a complete dataset independently, then results are merged.

## Setup

### 1. Server (Raspberry Pi or Always-On Machine)

```bash
# On your Raspberry Pi or server
cd FinAI/distributed
python server.py
```

Server will display:
```
Server address: http://192.168.1.100:8765
Workers can connect to: http://192.168.1.100:8765
```

### 2. Workers (Laptop, PC, Friend's PC)

On each worker machine:

```bash
# Clone FinAI repo
git clone https://github.com/your-username/FinAI.git
cd FinAI

# Install dependencies
pip install -r requirements.txt

# Start worker (replace SERVER_IP with your server's IP)
python distributed/worker.py --server http://192.168.1.100:8765
```

### 3. Submit Tasks

From any machine:

```bash
# Submit all pending datasets from datasets.csv
python distributed/client.py --server http://192.168.1.100:8765 submit

# Check status
python distributed/client.py --server http://192.168.1.100:8765 status

# View workers
python distributed/client.py --server http://192.168.1.100:8765 workers

# View tasks
python distributed/client.py --server http://192.168.1.100:8765 tasks
```

## Example Workflow

### Step 1: Start Server (Raspberry Pi)
```bash
python distributed/server.py
```

### Step 2: Start Workers (3 machines)

**Laptop:**
```bash
python distributed/worker.py --server http://192.168.1.100:8765
```

**Your PC:**
```bash
python distributed/worker.py --server http://192.168.1.100:8765
```

**Friend's PC:**
```bash
python distributed/worker.py --server http://192.168.1.100:8765 --worker-id friend_pc
```

### Step 3: Submit Tasks
```bash
python distributed/client.py --server http://192.168.1.100:8765 submit
```

Output:
```
Found 22 pending datasets

Submitted: yukiarimo/english-vocabulary (task_id: a1b2c3d4)
Submitted: vumichien/financial-sentiment (task_id: e5f6g7h8)
Submitted: TimKoornstra/financial-tweets-sentiment (task_id: i9j0k1l2)
...

Submitted 22/22 tasks
```

### Step 4: Monitor Progress
```bash
python distributed/client.py --server http://192.168.1.100:8765 status
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

## Performance

### Sequential (1 machine)
- 22 datasets × 2 hours each = **44 hours total**

### Distributed (3 workers)
- 22 datasets ÷ 3 workers ≈ 7-8 datasets per worker
- 8 datasets × 2 hours = **16 hours total** (2.75x faster)

### Distributed (4 workers)
- 22 datasets ÷ 4 workers ≈ 5-6 datasets per worker
- 6 datasets × 2 hours = **12 hours total** (3.67x faster)

## Network Requirements

- All machines must be on same network (or use VPN/port forwarding)
- Server needs static IP or hostname
- Firewall must allow port 8765
- Recommended: Gigabit LAN for faster model shard transfers

## Hardware Recommendations

### Server (Raspberry Pi)
- Raspberry Pi 4 (4GB+ RAM)
- 32GB+ SD card
- Ethernet connection (not WiFi)
- Always-on power

### Workers
- Any machine with Python 3.8+
- GPU recommended but not required
- 8GB+ RAM
- 10GB+ free disk space

## Troubleshooting

### Workers can't connect to server
```bash
# Check server is running
curl http://SERVER_IP:8765/status

# Check firewall (on server)
sudo ufw allow 8765

# Check network connectivity
ping SERVER_IP
```

### Worker keeps disconnecting
- Check network stability
- Ensure server is always-on
- Worker sends heartbeat every 30s
- Server removes workers after 5min of no heartbeat

### Tasks not being assigned
```bash
# Check task queue
python distributed/client.py --server http://SERVER_IP:8765 tasks

# Re-submit tasks if needed
python distributed/client.py --server http://SERVER_IP:8765 submit
```

## Advanced: Model Shard Merging

After all workers complete, you can merge model shards:

```bash
# TODO: Implement shard merging
python distributed/merge_shards.py --server http://SERVER_IP:8765
```

This will:
1. Download all model shards from workers
2. Average/merge weights
3. Save final unified model to `models/finai_gpt.pt`

## Security Notes

- Server has NO authentication (use on trusted network only)
- Do NOT expose server to public internet
- Use VPN for remote workers
- Consider adding API keys for production use

## Limitations

- This is task-level parallelism, NOT data parallelism
- Workers train independently (no gradient sharing)
- Final model quality depends on shard merging strategy
- Network latency affects task distribution speed

## Future Improvements

- [ ] Model shard merging implementation
- [ ] Authentication/API keys
- [ ] HTTPS support
- [ ] Web dashboard for monitoring
- [ ] Automatic worker discovery (mDNS/Bonjour)
- [ ] Resume interrupted tasks
- [ ] Priority queue for tasks
- [ ] Worker resource allocation (GPU/CPU)
