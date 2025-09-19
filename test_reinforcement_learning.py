#!/usr/bin/env python3
"""
Test Reinforcement Learning System
Test the reward-based learning system with historical data
"""

import asyncio
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.services.reinforcement_learning_service import ReinforcementLearningService

async def test_reinforcement_learning():
    """Test the reinforcement learning system"""
    print("MeridianAI Reinforcement Learning Test")
    print("=" * 50)
    
    # Initialize RL service
    rl_service = ReinforcementLearningService()
    await rl_service.initialize_learning_system()
    
    # Test companies
    test_companies = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"\nTesting reinforcement learning with {len(test_companies)} companies...")
    print("=" * 60)
    
    for symbol in test_companies:
        try:
            print(f"\nTesting {symbol}...")
            
            # Define training periods (monthly predictions)
            periods = [
                ("2023-01-01", "2023-12-31", "2023"),
                ("2024-01-01", "2024-12-31", "2024")
            ]
            
            total_score = 0
            for start_date, end_date, year in periods:
                print(f"  Training on {year} data...")
                
                start_time = time.time()
                await rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
                end_time = time.time()
                
                # Get current score
                current_score = rl_service.get_current_score()
                total_score = current_score
                
                print(f"    Score: {current_score:.2f}")
                print(f"    Training time: {end_time - start_time:.1f}s")
            
            print(f"  Final Score for {symbol}: {total_score:.2f}")
            
        except Exception as e:
            print(f"  Error testing {symbol}: {str(e)}")
            continue
    
    # Show final performance
    performance = rl_service.get_model_performance()
    
    print(f"\nFinal Performance Summary")
    print("=" * 30)
    print(f"Total Score: {performance['total_score']:.2f}")
    print(f"Max Score: {performance['max_score']}")
    print(f"Score Percentage: {(performance['total_score'] / performance['max_score']) * 100:.2f}%")
    
    print(f"\nModel Weights:")
    print("-" * 15)
    for model, weight in performance['model_weights'].items():
        print(f"{model}: {weight:.2f}")
    
    print(f"\nReinforcement Learning Test Completed!")

if __name__ == "__main__":
    asyncio.run(test_reinforcement_learning())
