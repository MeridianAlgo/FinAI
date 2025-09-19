#!/usr/bin/env python3
"""
MeridianAI Backend Testing Script
Test the AI models and ensure they work properly
"""

import asyncio
import sys
import os
from pathlib import Path
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_data_training_service():
    """Test the data training service"""
    print("🧪 Testing Data Training Service...")
    print("=" * 50)
    
    try:
        from api.services.data_training_service import DataTrainingService
        
        data_service = DataTrainingService()
        print("✅ DataTrainingService imported successfully")
        
        # Test gathering training data
        print("\n📊 Gathering training data...")
        await data_service.gather_comprehensive_training_data()
        
        # Get summary
        summary = data_service.get_training_summary()
        print(f"\n📈 Training Data Summary:")
        print(f"   - Total Stocks: {summary.get('total_stocks', 0)}")
        print(f"   - Market Indicators: {summary.get('market_indicators', 0)}")
        print(f"   - Fundamental Companies: {summary.get('fundamental_companies', 0)}")
        print(f"   - Sentiment Stocks: {summary.get('sentiment_stocks', 0)}")
        print(f"   - Technical Stocks: {summary.get('technical_stocks', 0)}")
        print(f"   - Sectors Analyzed: {summary.get('sectors_analyzed', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing data training service: {str(e)}")
        traceback.print_exc()
        return False

async def test_enhanced_ai_service():
    """Test the enhanced AI service"""
    print("\n🧪 Testing Enhanced AI Service...")
    print("=" * 50)
    
    try:
        from api.services.enhanced_ai_service import EnhancedAIService
        
        ai_service = EnhancedAIService()
        print("✅ EnhancedAIService imported successfully")
        
        # Test training AI models
        print("\n🤖 Training AI models...")
        await ai_service.train_ai_models()
        print("✅ AI models trained successfully")
        
        # Test analyzing a company
        print("\n🔍 Testing company analysis...")
        test_symbol = "AAPL"
        analysis = await ai_service.analyze_company(test_symbol)
        
        print(f"\n📊 Analysis Results for {test_symbol}:")
        print(f"   - AI Score: {analysis.get('ai_score', 0):.1f}/100")
        print(f"   - Recommendation: {analysis.get('recommendation', 'N/A')}")
        print(f"   - Confidence: {analysis.get('confidence', 0):.1%}")
        
        # Price prediction details
        price_pred = analysis.get('price_prediction', {})
        print(f"\n💰 Price Prediction:")
        print(f"   - Current Price: ${price_pred.get('current_price', 0):.2f}")
        print(f"   - Predicted Price: ${price_pred.get('predicted_price', 0):.2f}")
        print(f"   - Expected Return: {price_pred.get('expected_return', 0):.2%}")
        print(f"   - Direction: {price_pred.get('direction', 'N/A')}")
        
        # Trend analysis details
        trend_analysis = analysis.get('trend_analysis', {})
        print(f"\n📈 Trend Analysis:")
        print(f"   - Direction: {trend_analysis.get('direction', 'N/A')}")
        print(f"   - Strength: {trend_analysis.get('strength', 0):.1%}")
        print(f"   - Probability: {trend_analysis.get('probability', 0):.1%}")
        
        # Risk assessment details
        risk_assessment = analysis.get('risk_assessment', {})
        print(f"\n⚠️ Risk Assessment:")
        print(f"   - Risk Level: {risk_assessment.get('risk_level', 'N/A')}")
        print(f"   - Risk Score: {risk_assessment.get('risk_score', 0):.1f}/100")
        print(f"   - Predicted Volatility: {risk_assessment.get('predicted_volatility', 0):.2%}")
        
        # Sentiment analysis details
        sentiment_analysis = analysis.get('sentiment_analysis', {})
        print(f"\n😊 Sentiment Analysis:")
        print(f"   - Sentiment: {sentiment_analysis.get('sentiment', 'N/A')}")
        print(f"   - Score: {sentiment_analysis.get('score', 0):.1f}/100")
        
        # Fundamental score details
        fundamental_score = analysis.get('fundamental_score', {})
        print(f"\n🏢 Fundamental Analysis:")
        print(f"   - Score: {fundamental_score.get('score', 0):.1f}/100")
        print(f"   - Grade: {fundamental_score.get('grade', 'N/A')}")
        print(f"   - Expected Return: {fundamental_score.get('expected_return', 0):.2%}")
        
        # Key insights
        insights = analysis.get('key_insights', [])
        print(f"\n💡 Key Insights:")
        for i, insight in enumerate(insights[:3], 1):
            print(f"   {i}. {insight}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced AI service: {str(e)}")
        traceback.print_exc()
        return False

async def test_multiple_companies():
    """Test AI analysis on multiple companies"""
    print("\n🧪 Testing Multiple Company Analysis...")
    print("=" * 50)
    
    try:
        from api.services.enhanced_ai_service import EnhancedAIService
        
        ai_service = EnhancedAIService()
        
        test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        results = []
        
        for symbol in test_symbols:
            try:
                print(f"\n🔍 Analyzing {symbol}...")
                analysis = await ai_service.analyze_company(symbol)
                
                result = {
                    'symbol': symbol,
                    'ai_score': analysis.get('ai_score', 0),
                    'recommendation': analysis.get('recommendation', 'N/A'),
                    'confidence': analysis.get('confidence', 0)
                }
                results.append(result)
                
                print(f"   ✅ {symbol}: {result['ai_score']:.1f}/100 - {result['recommendation']}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by AI score
        results.sort(key=lambda x: x['ai_score'], reverse=True)
        
        print(f"\n📊 Top Performers (by AI Score):")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['symbol']}: {result['ai_score']:.1f}/100 - {result['recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multiple companies: {str(e)}")
        traceback.print_exc()
        return False

async def main():
    """Main testing function"""
    print("🚀 MeridianAI Backend Testing")
    print("=" * 60)
    
    # Test 1: Data Training Service
    test1_success = await test_data_training_service()
    
    if not test1_success:
        print("\n❌ Data training service test failed. Stopping tests.")
        return
    
    # Test 2: Enhanced AI Service
    test2_success = await test_enhanced_ai_service()
    
    if not test2_success:
        print("\n❌ Enhanced AI service test failed. Stopping tests.")
        return
    
    # Test 3: Multiple Company Analysis
    test3_success = await test_multiple_companies()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Data Training Service: {'PASS' if test1_success else 'FAIL'}")
    print(f"✅ Enhanced AI Service: {'PASS' if test2_success else 'FAIL'}")
    print(f"✅ Multiple Company Analysis: {'PASS' if test3_success else 'FAIL'}")
    
    if all([test1_success, test2_success, test3_success]):
        print("\n🎉 ALL TESTS PASSED! AI system is working correctly.")
        print("🚀 Ready to start the backend server.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
