#!/usr/bin/env python3
"""
MeridianAI Quick Test
Quick test of the AI system with a few companies
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.services.enhanced_ai_service import EnhancedAIService

async def quick_test():
    """Quick test of the AI system"""
    print("🚀 MeridianAI Quick Test")
    print("=" * 50)
    
    # Initialize AI service
    ai_service = EnhancedAIService()
    print("🤖 Training AI models...")
    await ai_service.train_ai_models()
    print("✅ AI models ready!")
    
    # Test companies
    test_companies = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
    
    print(f"\n📊 Testing {len(test_companies)} companies...")
    print("=" * 50)
    
    results = []
    for symbol in test_companies:
        try:
            print(f"\n🔍 Analyzing {symbol}...")
            analysis = await ai_service.analyze_company(symbol)
            
            result = {
                'symbol': symbol,
                'ai_score': analysis['ai_score'],
                'recommendation': analysis['recommendation'],
                'current_price': analysis['price_prediction']['current_price'],
                'expected_return': analysis['price_prediction']['expected_return'],
                'trend': analysis['trend_analysis']['direction'],
                'risk': analysis['risk_assessment']['risk_level'],
                'sentiment': analysis['sentiment_analysis']['sentiment']
            }
            results.append(result)
            
            print(f"   ✅ {symbol}: {result['ai_score']:.1f}/100 - {result['recommendation']}")
            print(f"      Price: ${result['current_price']:.2f} | Return: {result['expected_return']:.2%}")
            print(f"      Trend: {result['trend']} | Risk: {result['risk']} | Sentiment: {result['sentiment']}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {symbol}: {str(e)}")
    
    # Summary
    if results:
        results.sort(key=lambda x: x['ai_score'], reverse=True)
        
        print(f"\n🏆 TOP PERFORMERS:")
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['symbol']}: {result['ai_score']:.1f}/100 - {result['recommendation']}")
        
        avg_score = sum(r['ai_score'] for r in results) / len(results)
        print(f"\n📊 Average AI Score: {avg_score:.1f}/100")
        
        buy_recommendations = [r for r in results if 'Buy' in r['recommendation']]
        print(f"💡 Buy Recommendations: {len(buy_recommendations)}/{len(results)}")
    
    print("\n✅ Quick test completed!")

if __name__ == "__main__":
    asyncio.run(quick_test())
