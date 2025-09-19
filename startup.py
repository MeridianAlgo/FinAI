#!/usr/bin/env python3
"""
MeridianAI Startup Script
Initializes and trains AI models on startup
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.services.enhanced_ai_service import EnhancedAIService
from api.services.data_training_service import DataTrainingService

async def initialize_ai_system():
    """Initialize the AI system with comprehensive training"""
    print("🚀 Initializing MeridianAI Enhanced AI System...")
    print("=" * 60)
    
    try:
        # Initialize AI service
        ai_service = EnhancedAIService()
        
        print("📊 Gathering comprehensive training data...")
        # Gather training data first
        data_service = DataTrainingService()
        await data_service.gather_comprehensive_training_data()
        
        print("🤖 Training AI models with gathered data...")
        # Train AI models
        await ai_service.train_ai_models()
        
        print("✅ AI system initialization completed successfully!")
        print("=" * 60)
        print("🎯 AI Models Ready:")
        print(f"   - Price Prediction Model: ✅")
        print(f"   - Trend Analysis Model: ✅")
        print(f"   - Risk Assessment Model: ✅")
        print(f"   - Sentiment Analysis Model: ✅")
        print(f"   - Fundamental Scoring Model: ✅")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing AI system: {str(e)}")
        return False

async def main():
    """Main startup function"""
    print("🌟 MeridianAI Financial Analysis Platform")
    print("Enhanced AI System Startup")
    print("=" * 60)
    
    # Initialize AI system
    success = await initialize_ai_system()
    
    if success:
        print("🚀 Starting FastAPI server...")
        # Import and start the main application
        import uvicorn
        from main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    else:
        print("❌ Failed to initialize AI system. Exiting...")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
