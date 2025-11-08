# Web Dashboard Guide

## Overview

The FinAI Distributed Training Dashboard provides real-time monitoring of your distributed training system with:
- **Live ETA** to completion
- **Worker status** and activity
- **Task progress** tracking
- **Auto-updating** every 2 seconds
- **Beautiful UI** with color-coded status

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-socketio
```

### 2. Start the Dashboard

```bash
python distributed/dashboard.py --server http://192.168.1.100:8765 --password MeridianAlgo@TRAIN
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## Features

### 📊 Real-Time Statistics

**Top Stats Cards:**
- **Server Status**: Online/Offline
- **Active Workers**: Number of workers currently connected
- **Pending Tasks**: Tasks waiting to be processed
- **Completed Tasks**: Total tasks finished
- **ETA to Completion**: Estimated time until all tasks are done

### 👥 Worker Monitoring

**For Each Worker:**
- **Worker ID**: Unique identifier
- **Status**: Training (green) or Idle (yellow)
- **Current Task**: Dataset being trained
- **Task ETA**: Time remaining for current task
- **Last Seen**: When worker last checked in
- **Tasks Completed**: Total tasks finished by this worker
- **Total Time**: Cumulative training time
- **Errors**: Number of failed tasks
- **GPU Info**: GPU availability and device count

### 📋 Task Tracking

**Pending Tasks:**
- Task ID
- Dataset name
- Submission time

**Completed Tasks:**
- Task ID
- Dataset name
- Worker that completed it
- Status (success/failed)
- Training time
- Completion time

## Dashboard Options

### Custom Server URL

```bash
python distributed/dashboard.py --server http://YOUR_PUBLIC_IP:8765 --password MeridianAlgo@TRAIN
```

### Custom Port

```bash
python distributed/dashboard.py --port 8080
```

### Network Access

Allow other devices to access the dashboard:
```bash
python distributed/dashboard.py --host 0.0.0.0 --port 5000
```

Then access from any device on your network: `http://YOUR_IP:5000`

## How ETA is Calculated

### Overall ETA
1. Calculate average task time from completed tasks
2. Count active workers
3. Formula: `(Remaining Tasks × Average Time) / Active Workers`

### Worker Task ETA
- Tracks time since task was assigned
- Estimates based on average task time (default 2 hours)
- Updates in real-time

## Color Coding

### Worker Status
- **Green (Training)**: Worker is actively training a dataset
- **Yellow (Idle)**: Worker is connected but waiting for tasks

### Task Status
- **Green (Success)**: Task completed successfully
- **Red (Failed)**: Task failed with errors

## Connection Status

**Top-right indicator:**
- 🟢 **Connected**: Dashboard is receiving live updates
- 🔴 **Disconnected**: Connection lost, trying to reconnect

## Auto-Update

The dashboard automatically updates every **2 seconds** with:
- Latest worker status
- New completed tasks
- Updated ETA
- Current statistics

## Troubleshooting

### Dashboard Won't Start

**Error: Module not found**
```bash
pip install flask flask-socketio
```

**Error: Connection refused**
- Verify server is running
- Check server URL is correct
- Ensure password is correct

### No Data Showing

- Verify workers are connected
- Check tasks have been submitted
- Ensure authentication password is correct

### Dashboard Not Updating

- Check connection status (top-right)
- Refresh browser page
- Restart dashboard

## Example Usage

### Scenario: Monitoring 3 Workers

**Terminal 1 (Server):**
```bash
cd distributed
python server.py
```

**Terminal 2 (Dashboard):**
```bash
python dashboard.py --server http://localhost:8765 --password MeridianAlgo@TRAIN
```

**Browser:**
Open http://localhost:5000

**What You'll See:**
- 3 active workers
- Real-time task progress
- ETA counting down
- Workers switching between training and idle
- Completed tasks list growing

## Advanced Configuration

### Change Update Interval

Edit `dashboard.py` line 18:
```python
UPDATE_INTERVAL = 5  # Update every 5 seconds instead of 2
```

### Custom Styling

Edit the HTML template in `templates/dashboard.html` to customize colors, fonts, and layout.

## API Endpoints

The dashboard also provides a JSON API:

**Get Status:**
```bash
curl http://localhost:5000/api/status
```

Returns:
```json
{
  "server": {
    "status": "online",
    "timestamp": "2s ago"
  },
  "stats": {
    "workers": 3,
    "active_workers": 3,
    "pending_tasks": 19,
    "completed_tasks": 3,
    "eta": "14h 30m",
    "eta_seconds": 52200
  },
  "workers": [...],
  "pending_tasks": [...],
  "completed_tasks": [...]
}
```

## Tips

1. **Keep Dashboard Open**: Leave it running in a browser tab to monitor progress
2. **Mobile Access**: Access from phone/tablet using network IP
3. **Multiple Monitors**: Open dashboard on second monitor while working
4. **Screenshot Progress**: Take screenshots to track training over time
5. **Share with Team**: Send dashboard URL to collaborators

## Summary

The web dashboard provides:
- ✅ Real-time monitoring
- ✅ ETA calculations
- ✅ Worker status tracking
- ✅ Task progress
- ✅ Beautiful, responsive UI
- ✅ Auto-updating every 2 seconds
- ✅ Network accessible
- ✅ No configuration needed

**Start monitoring your distributed training in seconds!**

---
*Last updated: November 5, 2025*
