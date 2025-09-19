#!/usr/bin/env python3
"""
MeridianAI CLI Interface
Minimalistic command-line financial analysis tool
"""

import asyncio
import sys
import os
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import threading

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.services.enhanced_ai_service import EnhancedAIService

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

class MeridianCLI:
    def __init__(self):
        self.ai_service = None
        self.running = True
        
    async def initialize(self):
        """Initialize the AI service without training models"""
        print("MeridianAI Financial Analysis CLI")
        print("=" * 50)
        print("Initializing AI system...")
        
        try:
            self.ai_service = EnhancedAIService()
            print("AI system ready!")
            return True
        except Exception as e:
            print(f"Error initializing AI system: {str(e)}")
            return False
    
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "=" * 50)
        print("MERIDIANAI FINANCIAL ANALYSIS")
        print("=" * 50)
        print("1. Analyze Company")
        print("2. Compare Companies")
        print("3. Price Prediction")
        print("4. Trend Analysis")
        print("5. Risk Assessment")
        print("6. Market Overview")
        print("7. Train Models (Advanced)")
        print("0. Exit")
        print("=" * 50)
    
    async def analyze_company(self):
        """Analyze a single company with 10 years of data"""
        print("\nCompany Analysis")
        print("-" * 20)
        
        symbol = input("Enter stock symbol: ").upper().strip()
        if not symbol:
            print("Please enter a valid symbol")
            return
        
        try:
            # Show loading spinner
            spinner = LoadingSpinner(f"Analyzing {symbol} with 10 years of data")
            spinner.start()
            
            # Train models with extended data for this specific company
            await self._train_models_for_company(symbol)
            
            # Analyze the company
            analysis = await self.ai_service.analyze_company(symbol)
            spinner.stop()
            
            self._display_analysis_results(symbol, analysis)
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    async def _train_models_for_company(self, symbol: str):
        """Train models with extended data for a specific company"""
        try:
            # Get 10 years of data for the specific company
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Get 10 years of historical data
            hist_data = ticker.history(period="10y")
            if hist_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Get company info
            info = ticker.info
            
            # Calculate advanced technical indicators
            hist_data = self._calculate_advanced_indicators(hist_data)
            
            # Create training data with more samples
            training_data = self._create_extended_training_data(hist_data, info, symbol)
            
            # Train models with this extended data
            await self._train_models_with_data(training_data)
            
        except Exception as e:
            print(f"Error training models for {symbol}: {str(e)}")
            raise
    
    def _calculate_advanced_indicators(self, data):
        """Calculate advanced technical indicators"""
        import pandas as pd
        import numpy as np
        
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
        
        # Stochastic Oscillator
        low_14 = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        data['Williams_R'] = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
        
        # Average True Range
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['ATR'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price momentum
        data['Momentum_1'] = data['Close'].pct_change(1)
        data['Momentum_5'] = data['Close'].pct_change(5)
        data['Momentum_10'] = data['Close'].pct_change(10)
        data['Momentum_20'] = data['Close'].pct_change(20)
        
        # Volatility
        data['Volatility_10'] = data['Close'].pct_change().rolling(window=10).std()
        data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()
        data['Volatility_30'] = data['Close'].pct_change().rolling(window=30).std()
        
        return data
    
    def _create_extended_training_data(self, hist_data, info, symbol):
        """Create extended training data with more samples"""
        import pandas as pd
        import numpy as np
        
        all_data = []
        
        # Create more training samples from 10 years of data
        for i in range(200, len(hist_data) - 30):  # Leave 30 days for prediction
            sample = {
                'symbol': symbol,
                'date': hist_data.index[i],
                'price': hist_data['Close'].iloc[i],
                'volume': hist_data['Volume'].iloc[i],
                'sma_20': hist_data['SMA_20'].iloc[i] if 'SMA_20' in hist_data.columns else 0,
                'sma_50': hist_data['SMA_50'].iloc[i] if 'SMA_50' in hist_data.columns else 0,
                'rsi': hist_data['RSI'].iloc[i] if 'RSI' in hist_data.columns else 50,
                'macd': hist_data['MACD'].iloc[i] if 'MACD' in hist_data.columns else 0,
                'volatility': hist_data['Close'].iloc[i-20:i].std() if i >= 20 else 0,
                'price_change_1d': hist_data['Close'].pct_change().iloc[i],
                'price_change_7d': hist_data['Close'].pct_change(7).iloc[i],
                'price_change_30d': hist_data['Close'].pct_change(30).iloc[i],
                'volume_ratio': hist_data['Volume'].iloc[i] / hist_data['Volume'].iloc[i-20:i].mean() if i >= 20 else 1,
                'future_return_7d': hist_data['Close'].pct_change(7).iloc[i+7] if i+7 < len(hist_data) else 0,
                'future_return_30d': hist_data['Close'].pct_change(30).iloc[i+30] if i+30 < len(hist_data) else 0,
                'future_volatility': hist_data['Close'].iloc[i+1:i+31].std() if i+30 < len(hist_data) else 0,
                'market_cap': info.get('marketCap', 0) / 1e9,
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0),
                'sentiment_score': self._calculate_sentiment_score(hist_data, i)
            }
            all_data.append(sample)
        
        return pd.DataFrame(all_data)
    
    def _calculate_sentiment_score(self, data, index):
        """Calculate sentiment score based on price action"""
        try:
            if index < 20:
                return 50.0
            
            # Price momentum indicators
            price_momentum_7d = data['Close'].pct_change(7).iloc[index]
            volume_ratio = data['Volume'].iloc[index] / data['Volume'].iloc[index-20:index].mean()
            volatility = data['Close'].iloc[index-20:index].std() / data['Close'].iloc[index-20:index].mean()
            
            # Weighted sentiment calculation
            momentum_score = np.tanh(price_momentum_7d * 10) * 50 + 50
            volume_score = min(100, max(0, (volume_ratio - 1) * 50 + 50))
            volatility_score = max(0, 100 - volatility * 100)
            
            sentiment_score = (momentum_score * 0.5 + volume_score * 0.3 + volatility_score * 0.2)
            return max(0, min(100, sentiment_score))
        except:
            return 50.0
    
    async def _train_models_with_data(self, data):
        """Train models with the provided data"""
        try:
            # Train price prediction model
            await self._train_price_model(data)
            
            # Train trend analysis model
            await self._train_trend_model(data)
            
            # Train risk assessment model
            await self._train_risk_model(data)
            
            # Train sentiment analysis model
            await self._train_sentiment_model(data)
            
            # Train fundamental scoring model
            await self._train_fundamental_model(data)
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            raise
    
    async def _train_price_model(self, data):
        """Train price prediction model with improved accuracy"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np
            
            # Prepare features
            feature_columns = [
                'sma_20', 'sma_50', 'rsi', 'macd', 'volatility', 'price_change_1d',
                'price_change_7d', 'price_change_30d', 'volume_ratio', 'market_cap',
                'pe_ratio', 'beta', 'sentiment_score'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['future_return_30d'].fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Use more sophisticated model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.ai_service.models['price_predictor'] = model
            self.ai_service.trained_models['price_features'] = available_features
            self.ai_service.trained_models['price_scaler'] = scaler
            
        except Exception as e:
            print(f"Error training price model: {str(e)}")
    
    async def _train_trend_model(self, data):
        """Train trend analysis model"""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np
            
            feature_columns = [
                'sma_20', 'sma_50', 'rsi', 'macd', 'volatility', 'price_change_1d',
                'price_change_7d', 'volume_ratio', 'sentiment_score'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = (data['future_return_7d'] > 0).astype(int)
            
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            self.ai_service.models['trend_analyzer'] = model
            self.ai_service.trained_models['trend_features'] = available_features
            self.ai_service.trained_models['trend_scaler'] = scaler
            
        except Exception as e:
            print(f"Error training trend model: {str(e)}")
    
    async def _train_risk_model(self, data):
        """Train risk assessment model"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np
            
            feature_columns = [
                'volatility', 'beta', 'price_change_30d', 'volume_ratio', 
                'market_cap', 'pe_ratio', 'sentiment_score'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['future_volatility'].fillna(0)
            
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            self.ai_service.models['risk_assessor'] = model
            self.ai_service.trained_models['risk_features'] = available_features
            self.ai_service.trained_models['risk_scaler'] = scaler
            
        except Exception as e:
            print(f"Error training risk model: {str(e)}")
    
    async def _train_sentiment_model(self, data):
        """Train sentiment analysis model"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np
            
            feature_columns = [
                'price_change_1d', 'price_change_7d', 'price_change_30d',
                'volume_ratio', 'volatility', 'sentiment_score'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['sentiment_score'].fillna(50)
            
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 50)
            
            model = LinearRegression()
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            self.ai_service.models['sentiment_analyzer'] = model
            self.ai_service.trained_models['sentiment_features'] = available_features
            self.ai_service.trained_models['sentiment_scaler'] = scaler
            
        except Exception as e:
            print(f"Error training sentiment model: {str(e)}")
    
    async def _train_fundamental_model(self, data):
        """Train fundamental scoring model"""
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import MinMaxScaler
            import numpy as np
            
            feature_columns = [
                'pe_ratio', 'beta', 'market_cap', 'sentiment_score'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['future_return_30d'].fillna(0)
            
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            model = Ridge(alpha=1.0)
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            model.fit(X_scaled, y)
            
            self.ai_service.models['fundamental_scorer'] = model
            self.ai_service.trained_models['fundamental_features'] = available_features
            self.ai_service.trained_models['fundamental_scaler'] = scaler
            
        except Exception as e:
            print(f"Error training fundamental model: {str(e)}")
    
    def _display_analysis_results(self, symbol, analysis):
        """Display analysis results in a clean format"""
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
        
        # Key Insights
        print(f"\nKey Insights:")
        for i, insight in enumerate(analysis['key_insights'][:3], 1):
            print(f"  {i}. {insight}")
    
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
            spinner = LoadingSpinner("Comparing companies")
            spinner.start()
            
            comparisons = []
            for symbol in symbols:
                try:
                    await self._train_models_for_company(symbol)
                    analysis = await self.ai_service.analyze_company(symbol)
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
            
            await self._train_models_for_company(symbol)
            analysis = await self.ai_service.analyze_company(symbol)
            price_pred = analysis['price_prediction']
            
            spinner.stop()
            
            print(f"\nPrice Prediction for {symbol}")
            print("=" * 35)
            print(f"Current Price: ${price_pred['current_price']:.2f}")
            print(f"Predicted Price: ${price_pred['predicted_price']:.2f}")
            print(f"Expected Return: {price_pred['expected_return']:.2%}")
            print(f"Direction: {price_pred['direction']}")
            print(f"Confidence: {price_pred['confidence']:.1%}")
            
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
            
            await self._train_models_for_company(symbol)
            analysis = await self.ai_service.analyze_company(symbol)
            trend = analysis['trend_analysis']
            
            spinner.stop()
            
            print(f"\nTrend Analysis for {symbol}")
            print("=" * 30)
            print(f"Direction: {trend['direction']}")
            print(f"Strength: {trend['strength']:.1%}")
            print(f"Probability: {trend['probability']:.1%}")
            print(f"Confidence: {trend['confidence']:.1%}")
            
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
            
            await self._train_models_for_company(symbol)
            analysis = await self.ai_service.analyze_company(symbol)
            risk = analysis['risk_assessment']
            
            spinner.stop()
            
            print(f"\nRisk Assessment for {symbol}")
            print("=" * 30)
            print(f"Risk Level: {risk['risk_level']}")
            print(f"Risk Score: {risk['risk_score']:.1f}/100")
            print(f"Predicted Volatility: {risk['predicted_volatility']:.2%}")
            print(f"Confidence: {risk['confidence']:.1%}")
            
        except Exception as e:
            print(f"Error assessing risk for {symbol}: {str(e)}")
    
    async def market_overview(self):
        """Get market overview"""
        print("\nMarket Overview")
        print("-" * 15)
        
        try:
            spinner = LoadingSpinner("Analyzing market conditions")
            spinner.start()
            
            # Analyze major indices
            major_indices = ['^GSPC', '^DJI', '^IXIC']
            market_analysis = []
            
            for index in major_indices:
                try:
                    await self._train_models_for_company(index)
                    analysis = await self.ai_service.analyze_company(index)
                    market_analysis.append({
                        'index': index,
                        'ai_score': analysis['ai_score'],
                        'trend': analysis['trend_analysis']['direction'],
                        'risk': analysis['risk_assessment']['risk_level']
                    })
                except Exception:
                    continue
            
            spinner.stop()
            
            if market_analysis:
                avg_score = sum(ma['ai_score'] for ma in market_analysis) / len(market_analysis)
                bullish_count = sum(1 for ma in market_analysis if 'up' in ma['trend'])
                
                print(f"\nMarket Overview")
                print("=" * 20)
                print(f"Overall Score: {avg_score:.1f}/100")
                print(f"Market Sentiment: {'Bullish' if bullish_count > len(market_analysis) / 2 else 'Bearish'}")
                
                print(f"\nIndex Analysis:")
                print("-" * 20)
                for ma in market_analysis:
                    print(f"{ma['index']}: {ma['ai_score']:.1f}/100 | {ma['trend']} | {ma['risk']}")
            else:
                print("Unable to generate market overview")
            
        except Exception as e:
            print(f"Error generating market overview: {str(e)}")
    
    async def train_models_advanced(self):
        """Train models with comprehensive data"""
        print("\nAdvanced Model Training")
        print("-" * 25)
        
        confirm = input("This will train models with comprehensive market data. Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Training cancelled")
            return
        
        try:
            spinner = LoadingSpinner("Training models with comprehensive data")
            spinner.start()
            
            await self.ai_service.train_ai_models()
            
            spinner.stop()
            print("Models trained successfully!")
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
    
    async def run(self):
        """Main run loop"""
        if not await self.initialize():
            return
        
        while self.running:
            try:
                self.display_menu()
                choice = input("\nEnter your choice (0-7): ").strip()
                
                if choice == '0':
                    print("\nThank you for using MeridianAI!")
                    self.running = False
                elif choice == '1':
                    await self.analyze_company()
                elif choice == '2':
                    await self.compare_companies()
                elif choice == '3':
                    await self.price_prediction()
                elif choice == '4':
                    await self.trend_analysis()
                elif choice == '5':
                    await self.risk_assessment()
                elif choice == '6':
                    await self.market_overview()
                elif choice == '7':
                    await self.train_models_advanced()
                else:
                    print("Invalid choice. Please enter 0-7.")
                
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
    cli = MeridianCLI()
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
