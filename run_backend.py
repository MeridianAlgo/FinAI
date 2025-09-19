#!/usr/bin/env python3
"""
MeridianAI Backend Server
Run the FastAPI backend server with AI models
"""

import asyncio
import sys
import os
from pathlib import Path
import uvicorn

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import app

async def main():
    """Main function to run the backend server"""
    print("🚀 Starting MeridianAI Backend Server...")
    print("=" * 60)
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 AI Endpoints: http://localhost:8000/api/ai/")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
