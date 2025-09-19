#!/usr/bin/env python3
"""
MeridianAI Terminal Interface
Interactive terminal-based financial analysis tool
"""

import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.services.enhanced_ai_service import EnhancedAIService

class TerminalInterface:
    def __init__(self):
        self.ai_service = None
        self.running = True
        
    async def initialize(self):
        """Initialize the AI service"""
        print("🚀 Initializing MeridianAI Terminal Interface...")
        print("=" * 60)
        
        try:
            self.ai_service = EnhancedAIService()
            print("🤖 Training AI models...")
            await self.ai_service.train_ai_models()
            print("✅ AI system ready!")
            return True
        except Exception as e:
            print(f"❌ Error initializing AI system: {str(e)}")
            return False
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "=" * 60)
        print("🎯 MERIDIANAI FINANCIAL ANALYSIS TERMINAL")
        print("=" * 60)
        print("1. 📊 Analyze Company")
        print("2. 🔍 Compare Companies")
        print("3. 📈 Get AI Score")
        print("4. 💰 Price Prediction")
        print("5. 📊 Trend Analysis")
        print("6. ⚠️ Risk Assessment")
        print("7. 😊 Sentiment Analysis")
        print("8. 🏢 Fundamental Analysis")
        print("9. 📋 Market Insights")
        print("10. 🔄 Retrain Models")
        print("0. 🚪 Exit")
        print("=" * 60)
    
    async def analyze_company(self):
        """Analyze a single company"""
        print("\n📊 COMPANY ANALYSIS")
        print("-" * 30)
        
        symbol = input("Enter stock symbol (e.g., AAPL): ").upper().strip()
        if not symbol:
            print("❌ Please enter a valid symbol")
            return
        
        try:
            print(f"\n🔍 Analyzing {symbol}...")
            analysis = await self.ai_service.analyze_company(symbol)
            
            print(f"\n📊 ANALYSIS RESULTS FOR {symbol}")
            print("=" * 50)
            print(f"🎯 AI Score: {analysis['ai_score']:.1f}/100")
            print(f"💡 Recommendation: {analysis['recommendation']}")
            print(f"🎲 Confidence: {analysis['confidence']:.1%}")
            print(f"📅 Analysis Date: {analysis['analysis_date']}")
            
            # Price Prediction
            price_pred = analysis['price_prediction']
            print(f"\n💰 PRICE PREDICTION:")
            print(f"   Current Price: ${price_pred['current_price']:.2f}")
            print(f"   Predicted Price: ${price_pred['predicted_price']:.2f}")
            print(f"   Expected Return: {price_pred['expected_return']:.2%}")
            print(f"   Direction: {price_pred['direction']}")
            
            # Trend Analysis
            trend = analysis['trend_analysis']
            print(f"\n📈 TREND ANALYSIS:")
            print(f"   Direction: {trend['direction']}")
            print(f"   Strength: {trend['strength']:.1%}")
            print(f"   Probability: {trend['probability']:.1%}")
            
            # Risk Assessment
            risk = analysis['risk_assessment']
            print(f"\n⚠️ RISK ASSESSMENT:")
            print(f"   Risk Level: {risk['risk_level']}")
            print(f"   Risk Score: {risk['risk_score']:.1f}/100")
            print(f"   Predicted Volatility: {risk['predicted_volatility']:.2%}")
            
            # Sentiment Analysis
            sentiment = analysis['sentiment_analysis']
            print(f"\n😊 SENTIMENT ANALYSIS:")
            print(f"   Sentiment: {sentiment['sentiment']}")
            print(f"   Score: {sentiment['score']:.1f}/100")
            
            # Fundamental Analysis
            fundamental = analysis['fundamental_score']
            print(f"\n🏢 FUNDAMENTAL ANALYSIS:")
            print(f"   Score: {fundamental['score']:.1f}/100")
            print(f"   Grade: {fundamental['grade']}")
            print(f"   Expected Return: {fundamental['expected_return']:.2%}")
            
            # Key Insights
            print(f"\n💡 KEY INSIGHTS:")
            for i, insight in enumerate(analysis['key_insights'], 1):
                print(f"   {i}. {insight}")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
    
    async def compare_companies(self):
        """Compare multiple companies"""
        print("\n🔍 COMPANY COMPARISON")
        print("-" * 30)
        
        symbols_input = input("Enter stock symbols (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip()
        if not symbols_input:
            print("❌ Please enter valid symbols")
            return
        
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        if len(symbols) > 10:
            print("❌ Maximum 10 companies allowed")
            return
        
        try:
            print(f"\n🔍 Comparing {len(symbols)} companies...")
            
            comparisons = []
            for symbol in symbols:
                try:
                    print(f"   Analyzing {symbol}...")
                    analysis = await self.ai_service.analyze_company(symbol)
                    comparisons.append({
                        'symbol': symbol,
                        'ai_score': analysis['ai_score'],
                        'recommendation': analysis['recommendation'],
                        'confidence': analysis['confidence'],
                        'price_direction': analysis['price_prediction']['direction'],
                        'trend': analysis['trend_analysis']['direction'],
                        'risk': analysis['risk_assessment']['risk_level'],
                        'sentiment': analysis['sentiment_analysis']['sentiment'],
                        'grade': analysis['fundamental_score']['grade']
                    })
                except Exception as e:
                    print(f"   ❌ Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Sort by AI score
            comparisons.sort(key=lambda x: x['ai_score'], reverse=True)
            
            print(f"\n📊 COMPARISON RESULTS")
            print("=" * 80)
            print(f"{'Rank':<4} {'Symbol':<6} {'AI Score':<8} {'Recommend':<10} {'Trend':<12} {'Risk':<10} {'Grade':<4}")
            print("-" * 80)
            
            for i, comp in enumerate(comparisons, 1):
                print(f"{i:<4} {comp['symbol']:<6} {comp['ai_score']:<8.1f} {comp['recommendation']:<10} {comp['trend']:<12} {comp['risk']:<10} {comp['grade']:<4}")
            
            if comparisons:
                top_pick = comparisons[0]
                print(f"\n🏆 TOP PICK: {top_pick['symbol']}")
                print(f"   AI Score: {top_pick['ai_score']:.1f}/100")
                print(f"   Recommendation: {top_pick['recommendation']}")
                print(f"   Confidence: {top_pick['confidence']:.1%}")
            
        except Exception as e:
            print(f"❌ Error comparing companies: {str(e)}")
    
    async def get_ai_score(self):
        """Get AI score for a company"""
        print("\n📈 AI SCORE")
        print("-" * 20)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("❌ Please enter a valid symbol")
            return
        
        try:
            print(f"\n🔍 Getting AI score for {symbol}...")
            analysis = await self.ai_service.analyze_company(symbol)
            
            print(f"\n🎯 AI SCORE FOR {symbol}")
            print("=" * 30)
            print(f"Overall Score: {analysis['ai_score']:.1f}/100")
            print(f"Recommendation: {analysis['recommendation']}")
            print(f"Confidence: {analysis['confidence']:.1%}")
            
            # Score breakdown
            print(f"\n📊 SCORE BREAKDOWN:")
            print(f"   Price Prediction: {analysis['price_prediction']['expected_return']:.2%}")
            print(f"   Trend Analysis: {analysis['trend_analysis']['strength']:.1%}")
            print(f"   Risk Assessment: {100 - analysis['risk_assessment']['risk_score']:.1f}/100")
            print(f"   Sentiment: {analysis['sentiment_analysis']['score']:.1f}/100")
            print(f"   Fundamentals: {analysis['fundamental_score']['score']:.1f}/100")
            
        except Exception as e:
            print(f"❌ Error getting AI score for {symbol}: {str(e)}")
    
    async def price_prediction(self):
        """Get price prediction"""
        print("\n💰 PRICE PREDICTION")
        print("-" * 25)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("❌ Please enter a valid symbol")
            return
        
        try:
            print(f"\n🔍 Predicting price for {symbol}...")
            analysis = await self.ai_service.analyze_company(symbol)
            price_pred = analysis['price_prediction']
            
            print(f"\n💰 PRICE PREDICTION FOR {symbol}")
            print("=" * 40)
            print(f"Current Price: ${price_pred['current_price']:.2f}")
            print(f"Predicted Price: ${price_pred['predicted_price']:.2f}")
            print(f"Expected Return: {price_pred['expected_return']:.2%}")
            print(f"Direction: {price_pred['direction']}")
            print(f"Confidence: {price_pred['confidence']:.1%}")
            
            # Generate 30-day prediction
            current_price = price_pred['current_price']
            expected_return = price_pred['expected_return']
            
            print(f"\n📅 30-DAY PRICE FORECAST:")
            print("-" * 30)
            for day in [7, 14, 21, 30]:
                predicted_price = current_price * (1 + expected_return * (day / 30))
                change = predicted_price - current_price
                change_pct = (change / current_price) * 100
                print(f"Day {day:2d}: ${predicted_price:7.2f} ({change_pct:+.2f}%)")
            
        except Exception as e:
            print(f"❌ Error predicting price for {symbol}: {str(e)}")
    
    async def trend_analysis(self):
        """Get trend analysis"""
        print("\n📊 TREND ANALYSIS")
        print("-" * 25)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("❌ Please enter a valid symbol")
            return
        
        try:
            print(f"\n🔍 Analyzing trend for {symbol}...")
            analysis = await self.ai_service.analyze_company(symbol)
            trend = analysis['trend_analysis']
            
            print(f"\n📈 TREND ANALYSIS FOR {symbol}")
            print("=" * 35)
            print(f"Direction: {trend['direction']}")
            print(f"Strength: {trend['strength']:.1%}")
            print(f"Probability: {trend['probability']:.1%}")
            print(f"Confidence: {trend['confidence']:.1%}")
            
            # Trend interpretation
            direction = trend['direction']
            strength = trend['strength']
            
            print(f"\n💡 TREND INTERPRETATION:")
            if 'strong_uptrend' in direction:
                print("   🚀 Strong bullish momentum detected")
            elif 'uptrend' in direction:
                print("   📈 Bullish trend with moderate strength")
            elif 'strong_downtrend' in direction:
                print("   📉 Strong bearish momentum detected")
            elif 'downtrend' in direction:
                print("   📊 Bearish trend with moderate strength")
            else:
                print("   ➡️ Sideways movement, no clear trend")
            
            if strength > 0.7:
                print("   💪 High confidence in trend direction")
            elif strength > 0.5:
                print("   👍 Moderate confidence in trend direction")
            else:
                print("   🤔 Low confidence, trend may be weak")
            
        except Exception as e:
            print(f"❌ Error analyzing trend for {symbol}: {str(e)}")
    
    async def risk_assessment(self):
        """Get risk assessment"""
        print("\n⚠️ RISK ASSESSMENT")
        print("-" * 25)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("❌ Please enter a valid symbol")
            return
        
        try:
            print(f"\n🔍 Assessing risk for {symbol}...")
            analysis = await self.ai_service.analyze_company(symbol)
            risk = analysis['risk_assessment']
            
            print(f"\n⚠️ RISK ASSESSMENT FOR {symbol}")
            print("=" * 35)
            print(f"Risk Level: {risk['risk_level']}")
            print(f"Risk Score: {risk['risk_score']:.1f}/100")
            print(f"Predicted Volatility: {risk['predicted_volatility']:.2%}")
            print(f"Confidence: {risk['confidence']:.1%}")
            
            # Risk interpretation
            risk_level = risk['risk_level']
            risk_score = risk['risk_score']
            
            print(f"\n💡 RISK INTERPRETATION:")
            if risk_level in ['very_high', 'high']:
                print("   🔴 HIGH RISK - Consider position sizing carefully")
                print("   💡 Recommendations:")
                print("      - Use smaller position sizes")
                print("      - Set stop-loss orders")
                print("      - Monitor closely")
            elif risk_level == 'medium':
                print("   🟡 MEDIUM RISK - Standard risk management")
                print("   💡 Recommendations:")
                print("      - Normal position sizing")
                print("      - Regular monitoring")
            else:
                print("   🟢 LOW RISK - Suitable for conservative investors")
                print("   💡 Recommendations:")
                print("      - Can use larger position sizes")
                print("      - Less frequent monitoring needed")
            
        except Exception as e:
            print(f"❌ Error assessing risk for {symbol}: {str(e)}")
    
    async def sentiment_analysis(self):
        """Get sentiment analysis"""
        print("\n😊 SENTIMENT ANALYSIS")
        print("-" * 30)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("❌ Please enter a valid symbol")
            return
        
        try:
            print(f"\n🔍 Analyzing sentiment for {symbol}...")
            analysis = await self.ai_service.analyze_company(symbol)
            sentiment = analysis['sentiment_analysis']
            
            print(f"\n😊 SENTIMENT ANALYSIS FOR {symbol}")
            print("=" * 40)
            print(f"Sentiment: {sentiment['sentiment']}")
            print(f"Score: {sentiment['score']:.1f}/100")
            print(f"Confidence: {sentiment['confidence']:.1%}")
            
            # Sentiment interpretation
            sentiment_type = sentiment['sentiment']
            score = sentiment['score']
            
            print(f"\n💡 SENTIMENT INTERPRETATION:")
            if sentiment_type == 'very_positive':
                print("   😍 Very positive market sentiment")
                print("   📈 Strong bullish sentiment indicators")
            elif sentiment_type == 'positive':
                print("   😊 Positive market sentiment")
                print("   📊 Generally bullish sentiment")
            elif sentiment_type == 'neutral':
                print("   😐 Neutral market sentiment")
                print("   ➡️ Mixed or balanced sentiment")
            elif sentiment_type == 'negative':
                print("   😟 Negative market sentiment")
                print("   📉 Bearish sentiment indicators")
            else:
                print("   😰 Very negative market sentiment")
                print("   📉 Strong bearish sentiment")
            
        except Exception as e:
            print(f"❌ Error analyzing sentiment for {symbol}: {str(e)}")
    
    async def fundamental_analysis(self):
        """Get fundamental analysis"""
        print("\n🏢 FUNDAMENTAL ANALYSIS")
        print("-" * 30)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("❌ Please enter a valid symbol")
            return
        
        try:
            print(f"\n🔍 Analyzing fundamentals for {symbol}...")
            analysis = await self.ai_service.analyze_company(symbol)
            fundamental = analysis['fundamental_score']
            
            print(f"\n🏢 FUNDAMENTAL ANALYSIS FOR {symbol}")
            print("=" * 40)
            print(f"Score: {fundamental['score']:.1f}/100")
            print(f"Grade: {fundamental['grade']}")
            print(f"Expected Return: {fundamental['expected_return']:.2%}")
            print(f"Confidence: {fundamental['confidence']:.1%}")
            
            # Grade interpretation
            grade = fundamental['grade']
            score = fundamental['score']
            
            print(f"\n💡 FUNDAMENTAL INTERPRETATION:")
            if grade in ['A+', 'A']:
                print("   🏆 EXCELLENT fundamentals")
                print("   💡 Strong financial health and growth prospects")
            elif grade in ['B+', 'B']:
                print("   👍 GOOD fundamentals")
                print("   💡 Solid financial health with room for improvement")
            elif grade in ['C+', 'C']:
                print("   ⚖️ AVERAGE fundamentals")
                print("   💡 Mixed financial health, monitor closely")
            elif grade == 'D':
                print("   ⚠️ POOR fundamentals")
                print("   💡 Weak financial health, high risk")
            else:
                print("   🚨 VERY POOR fundamentals")
                print("   💡 Critical financial issues, avoid or exit")
            
        except Exception as e:
            print(f"❌ Error analyzing fundamentals for {symbol}: {str(e)}")
    
    async def market_insights(self):
        """Get market insights"""
        print("\n📋 MARKET INSIGHTS")
        print("-" * 25)
        
        try:
            print("🔍 Generating market insights...")
            
            # Analyze major indices
            major_indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
            market_analysis = []
            
            for index in major_indices:
                try:
                    print(f"   Analyzing {index}...")
                    analysis = await self.ai_service.analyze_company(index)
                    market_analysis.append({
                        'index': index,
                        'ai_score': analysis['ai_score'],
                        'trend': analysis['trend_analysis']['direction'],
                        'risk': analysis['risk_assessment']['risk_level'],
                        'sentiment': analysis['sentiment_analysis']['sentiment']
                    })
                except Exception:
                    continue
            
            if market_analysis:
                # Calculate overall market metrics
                avg_score = sum(ma['ai_score'] for ma in market_analysis) / len(market_analysis)
                bullish_count = sum(1 for ma in market_analysis if 'up' in ma['trend'])
                risk_levels = [ma['risk'] for ma in market_analysis]
                
                print(f"\n📊 MARKET INSIGHTS")
                print("=" * 30)
                print(f"Overall Market Score: {avg_score:.1f}/100")
                print(f"Market Sentiment: {'Bullish' if bullish_count > len(market_analysis) / 2 else 'Bearish'}")
                print(f"Dominant Risk Level: {max(set(risk_levels), key=risk_levels.count) if risk_levels else 'medium'}")
                
                print(f"\n📈 INDEX ANALYSIS:")
                print("-" * 25)
                for ma in market_analysis:
                    print(f"{ma['index']:<6}: Score {ma['ai_score']:5.1f} | {ma['trend']:<12} | {ma['risk']:<10} | {ma['sentiment']}")
                
                print(f"\n💡 MARKET RECOMMENDATIONS:")
                if avg_score > 70:
                    print("   🚀 Strong market conditions - consider increasing exposure")
                elif avg_score > 50:
                    print("   📊 Moderate market conditions - maintain current strategy")
                else:
                    print("   ⚠️ Weak market conditions - consider reducing exposure")
                
                if bullish_count > len(market_analysis) / 2:
                    print("   📈 Bullish trend detected across major indices")
                else:
                    print("   📉 Bearish trend detected across major indices")
            else:
                print("❌ Unable to generate market insights")
            
        except Exception as e:
            print(f"❌ Error generating market insights: {str(e)}")
    
    async def retrain_models(self):
        """Retrain AI models"""
        print("\n🔄 RETRAINING MODELS")
        print("-" * 25)
        
        confirm = input("This will retrain all AI models with latest data. Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Retraining cancelled")
            return
        
        try:
            print("🤖 Retraining AI models...")
            await self.ai_service.train_ai_models()
            print("✅ Models retrained successfully!")
        except Exception as e:
            print(f"❌ Error retraining models: {str(e)}")
    
    async def run(self):
        """Main run loop"""
        if not await self.initialize():
            return
        
        while self.running:
            try:
                self.display_menu()
                choice = input("\nEnter your choice (0-10): ").strip()
                
                if choice == '0':
                    print("\n👋 Thank you for using MeridianAI!")
                    self.running = False
                elif choice == '1':
                    await self.analyze_company()
                elif choice == '2':
                    await self.compare_companies()
                elif choice == '3':
                    await self.get_ai_score()
                elif choice == '4':
                    await self.price_prediction()
                elif choice == '5':
                    await self.trend_analysis()
                elif choice == '6':
                    await self.risk_assessment()
                elif choice == '7':
                    await self.sentiment_analysis()
                elif choice == '8':
                    await self.fundamental_analysis()
                elif choice == '9':
                    await self.market_insights()
                elif choice == '10':
                    await self.retrain_models()
                else:
                    print("❌ Invalid choice. Please enter 0-10.")
                
                if self.running:
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.running = False
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                input("Press Enter to continue...")

async def main():
    """Main function"""
    interface = TerminalInterface()
    await interface.run()

if __name__ == "__main__":
    asyncio.run(main())
