#!/usr/bin/env python3
"""
MeridianAI Financial Analysis Platform
Quick start script for development
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_node_version():
    """Check if Node.js is installed"""
    success, output = run_command("node --version")
    if not success:
        print("❌ Node.js is not installed. Please install Node.js 16 or higher")
        return False
    print(f"✅ Node.js {output.strip()} detected")
    return True

def install_backend_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    success, output = run_command("pip install -r requirements.txt")
    if not success:
        print(f"❌ Failed to install Python dependencies: {output}")
        return False
    print("✅ Python dependencies installed successfully")
    return True

def install_frontend_dependencies():
    """Install Node.js dependencies"""
    print("\n📦 Installing Node.js dependencies...")
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found")
        return False
    
    success, output = run_command("npm install", cwd=frontend_path)
    if not success:
        print(f"❌ Failed to install Node.js dependencies: {output}")
        return False
    print("✅ Node.js dependencies installed successfully")
    return True

def start_backend():
    """Start the backend server"""
    print("\n🚀 Starting backend server...")
    try:
        subprocess.Popen([sys.executable, "main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Backend server started on http://localhost:8000")
        return True
    except Exception as e:
        print(f"❌ Failed to start backend server: {e}")
        return False

def start_frontend():
    """Start the frontend server"""
    print("\n🚀 Starting frontend server...")
    frontend_path = Path("frontend")
    try:
        subprocess.Popen(["npm", "start"], cwd=frontend_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Frontend server started on http://localhost:3000")
        return True
    except Exception as e:
        print(f"❌ Failed to start frontend server: {e}")
        return False

def main():
    """Main function to set up and run the application"""
    print("🌟 MeridianAI Financial Analysis Platform")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        return 1
    
    if not check_node_version():
        return 1
    
    # Install dependencies
    if not install_backend_dependencies():
        return 1
    
    if not install_frontend_dependencies():
        return 1
    
    # Start servers
    if not start_backend():
        return 1
    
    time.sleep(2)  # Wait for backend to start
    
    if not start_frontend():
        return 1
    
    print("\n🎉 MeridianAI is now running!")
    print("=" * 50)
    print("📊 Backend API: http://localhost:8000")
    print("🌐 Frontend UI: http://localhost:3000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down MeridianAI...")
        return 0

if __name__ == "__main__":
    sys.exit(main())
