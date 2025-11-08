# Remote Access Setup Guide

This guide explains how to set up distributed training so workers from different networks (e.g., friends' homes) can connect to your server.

## Table of Contents
1. [Overview](#overview)
2. [Port Forwarding Setup](#port-forwarding-setup)
3. [Finding Your Public IP](#finding-your-public-ip)
4. [Router Configuration](#router-configuration)
5. [Firewall Configuration](#firewall-configuration)
6. [Testing Remote Access](#testing-remote-access)
7. [Security Best Practices](#security-best-practices)
8. [Dynamic DNS (Optional)](#dynamic-dns-optional)
9. [Troubleshooting](#troubleshooting)

## Overview

To allow workers from different networks to connect to your server, you need to:
1. Forward port 8765 on your router to your server's local IP
2. Configure your firewall to allow incoming connections
3. Share your public IP address with remote workers
4. Use the password `MeridianAlgo@TRAIN` for authentication

## Port Forwarding Setup

### Step 1: Find Your Server's Local IP

**On Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" (e.g., `192.168.1.100`)

**On Linux/Raspberry Pi:**
```bash
hostname -I
```
or
```bash
ip addr show
```

**On macOS:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

### Step 2: Access Your Router

1. Open a web browser
2. Go to your router's admin page (common addresses):
   - `http://192.168.1.1`
   - `http://192.168.0.1`
   - `http://10.0.0.1`
   - `http://router.asus.com` (ASUS routers)
   - `http://routerlogin.net` (Netgear routers)

3. Log in with your router credentials
   - Default credentials are often on a sticker on the router
   - Common defaults: admin/admin, admin/password

### Step 3: Configure Port Forwarding

Different routers have different interfaces, but the general steps are:

1. Find the **Port Forwarding** section (may be called):
   - Port Forwarding
   - Virtual Server
   - NAT Forwarding
   - Applications & Gaming

2. Create a new port forwarding rule:
   - **Service Name**: FinAI Server
   - **External Port**: 8765
   - **Internal Port**: 8765
   - **Internal IP**: Your server's local IP (e.g., 192.168.1.100)
   - **Protocol**: TCP
   - **Enabled**: Yes

3. Save the configuration

### Common Router Examples

#### Netgear Router
1. Advanced → Advanced Setup → Port Forwarding/Port Triggering
2. Click "Add Custom Service"
3. Fill in the details and click "Apply"

#### TP-Link Router
1. Forwarding → Virtual Servers
2. Click "Add New"
3. Fill in the details and click "Save"

#### ASUS Router
1. WAN → Virtual Server / Port Forwarding
2. Enable Port Forwarding
3. Add a new rule

#### Linksys Router
1. Applications & Gaming → Single Port Forwarding
2. Add a new application
3. Fill in the details and click "Save Settings"

## Finding Your Public IP

Your public IP is what remote workers will use to connect.

### Method 1: Online Services
Visit any of these websites:
- https://whatismyipaddress.com
- https://www.whatismyip.com
- https://ipinfo.io

### Method 2: Command Line

**Windows:**
```powershell
(Invoke-WebRequest -Uri "https://api.ipify.org").Content
```

**Linux/macOS:**
```bash
curl https://api.ipify.org
```

### Method 3: Router Admin Page
Most routers display your public IP on the main status page.

## Firewall Configuration

### Windows Firewall

1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Click "Inbound Rules" → "New Rule"
4. Select "Port" → Next
5. Select "TCP" and enter port 8765 → Next
6. Select "Allow the connection" → Next
7. Check all profiles (Domain, Private, Public) → Next
8. Name it "FinAI Server" → Finish

**Or use PowerShell (Run as Administrator):**
```powershell
New-NetFirewallRule -DisplayName "FinAI Server" -Direction Inbound -LocalPort 8765 -Protocol TCP -Action Allow
```

### Linux Firewall (UFW)

```bash
sudo ufw allow 8765/tcp
sudo ufw reload
sudo ufw status
```

### Linux Firewall (iptables)

```bash
sudo iptables -A INPUT -p tcp --dport 8765 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

### macOS Firewall

1. System Preferences → Security & Privacy → Firewall
2. Click "Firewall Options"
3. Click "+" to add an application
4. Select Python or your terminal app
5. Click "Add" → "OK"

## Testing Remote Access

### Test from Local Network

From another device on your network:
```bash
curl http://YOUR_LOCAL_IP:8765/status
```

Example:
```bash
curl http://192.168.1.100:8765/status
```

### Test from External Network

From a device on a different network (e.g., mobile hotspot):
```bash
curl http://YOUR_PUBLIC_IP:8765/status
```

Example:
```bash
curl http://203.0.113.45:8765/status
```

You should see:
```json
{"status": "online", "workers": 0, "pending_tasks": 0, "completed_tasks": 0, "timestamp": 1234567890.123, "auth_required": true}
```

## Security Best Practices

### 1. Strong Password
The default password is `MeridianAlgo@TRAIN`. To change it:

Edit `distributed/server_config.json`:
```json
{
  "auth_password": "YOUR_NEW_STRONG_PASSWORD"
}
```

### 2. IP Whitelisting (Optional)

If you know the public IPs of your workers, you can restrict access:

Edit `distributed/server_config.json`:
```json
{
  "allowed_ips": ["203.0.113.45", "198.51.100.67"]
}
```

### 3. Use HTTPS (Advanced)

For production use, consider setting up HTTPS with Let's Encrypt or a reverse proxy like nginx.

### 4. Monitor Logs

Regularly check server logs for unauthorized access attempts:
```bash
tail -f distributed_data/logs/server.log
```

### 5. Disable When Not in Use

Stop the server when not training:
```bash
# Press Ctrl+C to stop the server
```

## Dynamic DNS (Optional)

If your public IP changes frequently, use a Dynamic DNS service:

### Popular DDNS Services
- No-IP (https://www.noip.com)
- DuckDNS (https://www.duckdns.org)
- Dynu (https://www.dynu.com)

### Setup Example (DuckDNS)

1. Create a free account at https://www.duckdns.org
2. Create a subdomain (e.g., `myfinai.duckdns.org`)
3. Install the DuckDNS update client on your server
4. Workers can connect using `http://myfinai.duckdns.org:8765`

## Troubleshooting

### Workers Can't Connect

1. **Check port forwarding is active:**
   ```bash
   # From external network
   telnet YOUR_PUBLIC_IP 8765
   ```

2. **Check firewall:**
   ```bash
   # Windows
   netstat -an | findstr :8765
   
   # Linux
   sudo netstat -tulpn | grep :8765
   ```

3. **Verify server is running:**
   ```bash
   curl http://localhost:8765/status
   ```

4. **Check router logs:**
   - Look for blocked connections in your router's admin panel

### Connection Timeout

- Ensure your ISP doesn't block port 8765
- Try a different port (e.g., 8080, 8888)
- Check if your ISP uses CGNAT (Carrier-Grade NAT)

### Authentication Errors

- Verify password matches in:
  - `distributed/server_config.json`
  - Worker `--password` argument
  - Client `--password` argument

### Slow Performance

- Check network bandwidth
- Use wired Ethernet instead of WiFi
- Ensure no other heavy downloads/uploads

## Complete Setup Example

### Server (Your Home)

1. **Start server:**
   ```bash
   cd FinAI
   python distributed/server.py
   ```

2. **Note the addresses:**
   - Local: `http://192.168.1.100:8765`
   - Public: `http://203.0.113.45:8765`

### Worker (Friend's Home)

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/FinAI.git
   cd FinAI
   pip install -r requirements.txt
   ```

2. **Start worker:**
   ```bash
   python distributed/worker.py --server http://203.0.113.45:8765 --password MeridianAlgo@TRAIN
   ```

### Submit Tasks (Any Location)

```bash
python distributed/client.py --server http://203.0.113.45:8765 --password MeridianAlgo@TRAIN submit
```

## Network Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Internet                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   Your Home       Friend's Home   Cloud Worker
        │               │               │
   ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
   │ Router  │     │ Router  │     │         │
   │ Port    │     │         │     │         │
   │ Forward │     │         │     │         │
   │ :8765   │     │         │     │         │
   └────┬────┘     └────┬────┘     └────┬────┘
        │               │               │
   ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
   │ Server  │     │ Worker  │     │ Worker  │
   │ :8765   │     │         │     │         │
   └─────────┘     └─────────┘     └─────────┘
```

## Summary

1. **Server Setup:**
   - Forward port 8765 on your router
   - Configure firewall to allow port 8765
   - Start server with authentication enabled

2. **Worker Setup:**
   - Use your public IP address
   - Include password in connection command
   - Test connection before submitting tasks

3. **Security:**
   - Use strong password (`MeridianAlgo@TRAIN`)
   - Monitor server logs
   - Only share credentials with trusted users

---
*Last updated: November 5, 2025*
