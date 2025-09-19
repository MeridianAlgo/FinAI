#!/usr/bin/env python3
"""
MeridianAI Advanced CLI with Reinforcement Learning
Advanced command-line financial analysis with reward-based learning
"""

import asyncio
import sys
import os
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import threading
from datetime import datetime, timedelta

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.services.reinforcement_learning_service import ReinforcementLearningService

class LoadingSpinner:
    def __init__(self, message="Loading"):
        self.message = message
        self.spinner_chars = "|/-\\"
        self.running = False
        self.thread = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print(f"\r{self.message}... Done!     ")
    
    def _spin(self):
        i = 0
        while self.running:
            print(f"\r{self.message}... {self.spinner_chars[i % len(self.spinner_chars)]}", end="", flush=True)
            time.sleep(0.1)
            i += 1

class AdvancedMeridianCLI:
    def __init__(self):
        self.rl_service = None
        self.running = True
        
    async def initialize(self):
        """Initialize the reinforcement learning system"""
        print("MeridianAI Advanced CLI with Reinforcement Learning")
        print("=" * 60)
        print("Initializing AI system with reward-based learning...")
        
        try:
            self.rl_service = ReinforcementLearningService()
            await self.rl_service.initialize_learning_system()
            print("AI system with reinforcement learning ready!")
            return True
        except Exception as e:
            print(f"Error initializing AI system: {str(e)}")
            return False
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "=" * 60)
        print("MERIDIANAI ADVANCED FINANCIAL ANALYSIS")
        print("=" * 60)
        print("1. Analyze Company (with RL)")
        print("2. Train with Historical Data")
        print("3. Compare Companies")
        print("4. Price Prediction")
        print("5. Trend Analysis")
        print("6. Risk Assessment")
        print("7. Sentiment Analysis")
        print("8. Model Performance")
        print("9. Reinforcement Learning Status")
        print("10. Train on Multiple Periods")
        print("0. Exit")
        print("=" * 60)
    
    async def analyze_company_with_rl(self):
        """Analyze a company using reinforcement learning"""
        print("\nCompany Analysis with Reinforcement Learning")
        print("-" * 50)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        try:
            # Show loading spinner
            spinner = LoadingSpinner(f"Analyzing {symbol} with reinforcement learning")
            spinner.start()
            
            # Train with recent historical data first
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
            
            # Get current analysis
            analysis = await self._get_current_analysis(symbol)
            spinner.stop()
            
            self._display_analysis_results(symbol, analysis)
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    async def train_with_historical_data(self):
        """Train models with historical data"""
        print("\nTrain with Historical Data")
        print("-" * 30)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        
        if not start_date or not end_date:
            print("Please enter valid dates")
            return
        
        try:
            spinner = LoadingSpinner(f"Training {symbol} with historical data from {start_date} to {end_date}")
            spinner.start()
            
            await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
            
            spinner.stop()
            
            # Show training results
            performance = self.rl_service.get_model_performance()
            print(f"\nTraining completed for {symbol}")
            print(f"Total Score: {performance['total_score']:.2f}")
            print(f"Max Score: {performance['max_score']}")
            print(f"Model Weights: {performance['model_weights']}")
            
        except Exception as e:
            print(f"Error training with historical data: {str(e)}")
    
    async def train_on_multiple_periods(self):
        """Train on multiple historical periods"""
        print("\nTrain on Multiple Historical Periods")
        print("-" * 40)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        try:
            # Define multiple training periods
            periods = [
                ("2020-01-01", "2020-12-31", "2020"),
                ("2021-01-01", "2021-12-31", "2021"),
                ("2022-01-01", "2022-12-31", "2022"),
                ("2023-01-01", "2023-12-31", "2023"),
                ("2024-01-01", "2024-12-31", "2024")
            ]
            
            print(f"Training {symbol} on {len(periods)} historical periods...")
            
            total_score = 0
            for start_date, end_date, year in periods:
                spinner = LoadingSpinner(f"Training on {year} data")
                spinner.start()
                
                await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
                
                spinner.stop()
                
                # Get current score
                current_score = self.rl_service.get_current_score()
                total_score = current_score
                
                print(f"  {year}: Score = {current_score:.2f}")
            
            print(f"\nMulti-period training completed for {symbol}")
            print(f"Final Total Score: {total_score:.2f}")
            print(f"Max Possible Score: {self.rl_service.max_score}")
            print(f"Score Percentage: {(total_score / self.rl_service.max_score) * 100:.2f}%")
            
        except Exception as e:
            print(f"Error training on multiple periods: {str(e)}")
    
    async def _get_current_analysis(self, symbol: str) -> Dict:
        """Get current analysis for a symbol"""
        try:
            # This would integrate with the existing analysis system
            # For now, return a mock analysis
            return {
                'symbol': symbol,
                'ai_score': 75.5,
                'recommendation': 'Buy',
                'confidence': 0.85,
                'price_prediction': {
                    'current_price': 150.0,
                    'predicted_price': 155.0,
                    'expected_return': 0.033,
                    'direction': 'bullish'
                },
                'trend_analysis': {
                    'direction': 'uptrend',
                    'strength': 0.75,
                    'probability': 0.68
                },
                'risk_assessment': {
                    'risk_level': 'medium',
                    'risk_score': 35.0,
                    'predicted_volatility': 0.18
                },
                'sentiment_analysis': {
                    'sentiment': 'positive',
                    'score': 78.0
                },
                'fundamental_score': {
                    'score': 82.0,
                    'grade': 'A',
                    'expected_return': 0.042
                }
            }
        except Exception as e:
            print(f"Error getting current analysis: {str(e)}")
            return {}
    
    def _display_analysis_results(self, symbol: str, analysis: Dict):
        """Display analysis results"""
        if not analysis:
            print("No analysis data available")
            return
        
        print(f"\nAnalysis Results for {symbol}")
        print("=" * 40)
        print(f"AI Score: {analysis['ai_score']:.1f}/100")
        print(f"Recommendation: {analysis['recommendation']}")
        print(f"Confidence: {analysis['confidence']:.1%}")
        
        # Price Prediction
        price_pred = analysis['price_prediction']
        print(f"\nPrice Prediction:")
        print(f"  Current: ${price_pred['current_price']:.2f}")
        print(f"  Predicted: ${price_pred['predicted_price']:.2f}")
        print(f"  Expected Return: {price_pred['expected_return']:.2%}")
        print(f"  Direction: {price_pred['direction']}")
        
        # Trend Analysis
        trend = analysis['trend_analysis']
        print(f"\nTrend Analysis:")
        print(f"  Direction: {trend['direction']}")
        print(f"  Strength: {trend['strength']:.1%}")
        print(f"  Probability: {trend['probability']:.1%}")
        
        # Risk Assessment
        risk = analysis['risk_assessment']
        print(f"\nRisk Assessment:")
        print(f"  Level: {risk['risk_level']}")
        print(f"  Score: {risk['risk_score']:.1f}/100")
        print(f"  Volatility: {risk['predicted_volatility']:.2%}")
        
        # Sentiment Analysis
        sentiment = analysis['sentiment_analysis']
        print(f"\nSentiment Analysis:")
        print(f"  Sentiment: {sentiment['sentiment']}")
        print(f"  Score: {sentiment['score']:.1f}/100")
        
        # Fundamental Analysis
        fundamental = analysis['fundamental_score']
        print(f"\nFundamental Analysis:")
        print(f"  Score: {fundamental['score']:.1f}/100")
        print(f"  Grade: {fundamental['grade']}")
        print(f"  Expected Return: {fundamental['expected_return']:.2%}")
    
    async def compare_companies(self):
        """Compare multiple companies"""
        print("\nCompany Comparison")
        print("-" * 20)
        
        symbols_input = input("Enter stock symbols (comma-separated): ").strip()
        if not symbols_input:
            print("Please enter valid symbols")
            return
        
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        if len(symbols) > 5:
            print("Maximum 5 companies allowed")
            return
        
        try:
            spinner = LoadingSpinner("Comparing companies with reinforcement learning")
            spinner.start()
            
            comparisons = []
            for symbol in symbols:
                try:
                    # Train with recent data
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                    
                    await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
                    
                    analysis = await self._get_current_analysis(symbol)
                    comparisons.append({
                        'symbol': symbol,
                        'ai_score': analysis['ai_score'],
                        'recommendation': analysis['recommendation'],
                        'current_price': analysis['price_prediction']['current_price'],
                        'expected_return': analysis['price_prediction']['expected_return'],
                        'trend': analysis['trend_analysis']['direction'],
                        'risk': analysis['risk_assessment']['risk_level']
                    })
                except Exception as e:
                    print(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            spinner.stop()
            
            # Sort by AI score
            comparisons.sort(key=lambda x: x['ai_score'], reverse=True)
            
            print(f"\nComparison Results")
            print("=" * 60)
            print(f"{'Rank':<4} {'Symbol':<6} {'Score':<6} {'Recommend':<10} {'Price':<8} {'Return':<8}")
            print("-" * 60)
            
            for i, comp in enumerate(comparisons, 1):
                print(f"{i:<4} {comp['symbol']:<6} {comp['ai_score']:<6.1f} {comp['recommendation']:<10} ${comp['current_price']:<7.2f} {comp['expected_return']:<7.2%}")
            
            if comparisons:
                top_pick = comparisons[0]
                print(f"\nTop Pick: {top_pick['symbol']}")
                print(f"Score: {top_pick['ai_score']:.1f}/100")
                print(f"Recommendation: {top_pick['recommendation']}")
            
        except Exception as e:
            print(f"Error comparing companies: {str(e)}")
    
    async def price_prediction(self):
        """Get detailed price prediction"""
        print("\nPrice Prediction")
        print("-" * 20)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        try:
            spinner = LoadingSpinner(f"Generating price prediction for {symbol}")
            spinner.start()
            
            # Train with recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
            
            analysis = await self._get_current_analysis(symbol)
            price_pred = analysis['price_prediction']
            
            spinner.stop()
            
            print(f"\nPrice Prediction for {symbol}")
            print("=" * 35)
            print(f"Current Price: ${price_pred['current_price']:.2f}")
            print(f"Predicted Price: ${price_pred['predicted_price']:.2f}")
            print(f"Expected Return: {price_pred['expected_return']:.2%}")
            print(f"Direction: {price_pred['direction']}")
            
            # Generate 30-day forecast
            current_price = price_pred['current_price']
            expected_return = price_pred['expected_return']
            
            print(f"\n30-Day Price Forecast:")
            print("-" * 25)
            for day in [7, 14, 21, 30]:
                predicted_price = current_price * (1 + expected_return * (day / 30))
                change = predicted_price - current_price
                change_pct = (change / current_price) * 100
                print(f"Day {day:2d}: ${predicted_price:7.2f} ({change_pct:+.2f}%)")
            
        except Exception as e:
            print(f"Error predicting price for {symbol}: {str(e)}")
    
    async def trend_analysis(self):
        """Get trend analysis"""
        print("\nTrend Analysis")
        print("-" * 15)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        try:
            spinner = LoadingSpinner(f"Analyzing trend for {symbol}")
            spinner.start()
            
            # Train with recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
            
            analysis = await self._get_current_analysis(symbol)
            trend = analysis['trend_analysis']
            
            spinner.stop()
            
            print(f"\nTrend Analysis for {symbol}")
            print("=" * 30)
            print(f"Direction: {trend['direction']}")
            print(f"Strength: {trend['strength']:.1%}")
            print(f"Probability: {trend['probability']:.1%}")
            
        except Exception as e:
            print(f"Error analyzing trend for {symbol}: {str(e)}")
    
    async def risk_assessment(self):
        """Get risk assessment"""
        print("\nRisk Assessment")
        print("-" * 15)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        try:
            spinner = LoadingSpinner(f"Assessing risk for {symbol}")
            spinner.start()
            
            # Train with recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
            
            analysis = await self._get_current_analysis(symbol)
            risk = analysis['risk_assessment']
            
            spinner.stop()
            
            print(f"\nRisk Assessment for {symbol}")
            print("=" * 30)
            print(f"Risk Level: {risk['risk_level']}")
            print(f"Risk Score: {risk['risk_score']:.1f}/100")
            print(f"Predicted Volatility: {risk['predicted_volatility']:.2%}")
            
        except Exception as e:
            print(f"Error assessing risk for {symbol}: {str(e)}")
    
    async def sentiment_analysis(self):
        """Get sentiment analysis"""
        print("\nSentiment Analysis")
        print("-" * 20)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        try:
            spinner = LoadingSpinner(f"Analyzing sentiment for {symbol}")
            spinner.start()
            
            # Train with recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            await self.rl_service.train_with_reinforcement_learning(symbol, start_date, end_date)
            
            analysis = await self._get_current_analysis(symbol)
            sentiment = analysis['sentiment_analysis']
            
            spinner.stop()
            
            print(f"\nSentiment Analysis for {symbol}")
            print("=" * 35)
            print(f"Sentiment: {sentiment['sentiment']}")
            print(f"Score: {sentiment['score']:.1f}/100")
            
        except Exception as e:
            print(f"Error analyzing sentiment for {symbol}: {str(e)}")
    
    def model_performance(self):
        """Show model performance"""
        print("\nModel Performance")
        print("-" * 20)
        
        try:
            performance = self.rl_service.get_model_performance()
            
            print(f"Total Score: {performance['total_score']:.2f}")
            print(f"Max Score: {performance['max_score']}")
            print(f"Score Percentage: {(performance['total_score'] / performance['max_score']) * 100:.2f}%")
            
            print(f"\nModel Weights:")
            print("-" * 15)
            for model, weight in performance['model_weights'].items():
                print(f"{model}: {weight:.2f}")
            
            if performance['performance_data']:
                print(f"\nPerformance History:")
                print("-" * 20)
                for symbol, data in performance['performance_data'].items():
                    if data:
                        latest = data[-1]
                        print(f"{symbol}: {latest.get('total_reward', 0):.2f} (Latest)")
            
        except Exception as e:
            print(f"Error getting model performance: {str(e)}")
    
    def reinforcement_learning_status(self):
        """Show reinforcement learning status"""
        print("\nReinforcement Learning Status")
        print("-" * 35)
        
        try:
            performance = self.rl_service.get_model_performance()
            
            print(f"Current Total Score: {performance['total_score']:.2f}")
            print(f"Maximum Possible Score: {performance['max_score']}")
            print(f"Score Percentage: {(performance['total_score'] / performance['max_score']) * 100:.2f}%")
            
            print(f"\nLearning Progress:")
            print("-" * 20)
            if performance['total_score'] > 0:
                print("Status: Learning and improving")
                print("Models are being rewarded for accurate predictions")
            else:
                print("Status: Initializing")
                print("Models need more training data")
            
            print(f"\nModel Performance:")
            print("-" * 20)
            for model, weight in performance['model_weights'].items():
                status = "Good" if weight > 0 else "Needs Training"
                print(f"{model}: {weight:.2f} ({status})")
            
        except Exception as e:
            print(f"Error getting RL status: {str(e)}")
    
    async def run(self):
        """Main run loop"""
        if not await self.initialize():
            return
        
        while self.running:
            try:
                self.display_menu()
                choice = input("\nEnter your choice (0-10): ").strip()
                
                if choice == '0':
                    print("\nThank you for using MeridianAI Advanced CLI!")
                    self.running = False
                elif choice == '1':
                    await self.analyze_company_with_rl()
                elif choice == '2':
                    await self.train_with_historical_data()
                elif choice == '3':
                    await self.compare_companies()
                elif choice == '4':
                    await self.price_prediction()
                elif choice == '5':
                    await self.trend_analysis()
                elif choice == '6':
                    await self.risk_assessment()
                elif choice == '7':
                    await self.sentiment_analysis()
                elif choice == '8':
                    self.model_performance()
                elif choice == '9':
                    self.reinforcement_learning_status()
                elif choice == '10':
                    await self.train_on_multiple_periods()
                else:
                    print("Invalid choice. Please enter 0-10.")
                
                if self.running:
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                self.running = False
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                input("Press Enter to continue...")

async def main():
    """Main function"""
    cli = AdvancedMeridianCLI()
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
