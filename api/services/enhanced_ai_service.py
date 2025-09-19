import pandas as pd
import numpy as np
import asyncio
import aiohttp
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedAIService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.company_database = {}
        self.market_insights = {}
        self.trained_models = {}
        self.initialize_ai_models()
    
    def initialize_ai_models(self):
        """Initialize and train AI models with enhanced capabilities"""
        # Initialize multiple models for ensemble learning
        self.models = {
            'price_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'trend_analyzer': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'risk_assessor': RandomForestRegressor(n_estimators=50, random_state=42),
            'sentiment_analyzer': LinearRegression(),
            'fundamental_scorer': Ridge(alpha=1.0)
        }
        
        # Initialize scalers
        self.scalers = {
            'price_scaler': StandardScaler(),
            'feature_scaler': MinMaxScaler(),
            'sentiment_scaler': StandardScaler()
        }
    
    async def train_ai_models(self):
        """Train AI models with comprehensive market data"""
        try:
            # Get training data from multiple sources
            training_data = await self._gather_training_data()
            
            # Train price prediction model
            await self._train_price_model(training_data)
            
            # Train trend analysis model
            await self._train_trend_model(training_data)
            
            # Train risk assessment model
            await self._train_risk_model(training_data)
            
            # Train sentiment analysis model
            await self._train_sentiment_model(training_data)
            
            # Train fundamental scoring model
            await self._train_fundamental_model(training_data)
            
            print("✅ All AI models trained successfully")
            
        except Exception as e:
            print(f"❌ Error training AI models: {str(e)}")
    
    async def _gather_training_data(self) -> pd.DataFrame:
        """Gather comprehensive training data from multiple sources"""
        print("📊 Gathering training data...")
        
        # Use the data training service to gather comprehensive data
        from .data_training_service import DataTrainingService
        data_service = DataTrainingService()
        await data_service.gather_comprehensive_training_data()
        
        # Extract stock data for training
        stock_data = data_service.training_data.get('stock_data', {})
        
        # Major stock symbols for training
        symbols = list(stock_data.keys())[:50]  # Use first 50 stocks from gathered data
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Use pre-gathered data if available
                if symbol in stock_data:
                    hist_data = stock_data[symbol]['historical_data']
                    info = stock_data[symbol]['company_info']
                else:
                    # Fallback to direct API call
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(period="2y")
                    info = ticker.info
                
                if hist_data.empty:
                    continue
                
                # Calculate technical indicators
                hist_data = self._calculate_technical_indicators(hist_data)
                
                # Calculate fundamental metrics
                fundamental_metrics = self._calculate_fundamental_metrics(info, hist_data)
                
                # Calculate sentiment metrics (simulated based on price action)
                sentiment_metrics = self._calculate_sentiment_metrics(hist_data)
                
                # Create training samples
                for i in range(50, len(hist_data) - 30):  # Leave 30 days for prediction
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
                        **fundamental_metrics,
                        **sentiment_metrics
                    }
                    all_data.append(sample)
                
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(all_data)
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            # Moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
            
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
            
            return data
        except Exception as e:
            print(f"⚠️ Error calculating technical indicators: {str(e)}")
            return data
    
    def _calculate_fundamental_metrics(self, info: Dict, data: pd.DataFrame) -> Dict:
        """Calculate fundamental analysis metrics"""
        try:
            return {
                'market_cap': info.get('marketCap', 0) / 1e9,  # Billions
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'gross_margin': info.get('grossMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'profit_margin': info.get('profitMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'return_on_assets': info.get('returnOnAssets', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'enterprise_value': info.get('enterpriseValue', 0) / 1e9,  # Billions
                'free_cashflow': info.get('freeCashflow', 0) / 1e9,  # Billions
                'total_cash': info.get('totalCash', 0) / 1e9,  # Billions
                'total_debt': info.get('totalDebt', 0) / 1e9,  # Billions
                'book_value': info.get('bookValue', 0),
                'price_to_cashflow': info.get('priceToCashflow', 0),
                'ev_to_revenue': info.get('enterpriseToRevenue', 0),
                'ev_to_ebitda': info.get('enterpriseToEbitda', 0)
            }
        except Exception as e:
            print(f"⚠️ Error calculating fundamental metrics: {str(e)}")
            return {}
    
    def _calculate_sentiment_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate sentiment metrics based on price action and volume"""
        try:
            # Price momentum indicators
            price_momentum_1d = data['Close'].pct_change()
            price_momentum_7d = data['Close'].pct_change(7)
            price_momentum_30d = data['Close'].pct_change(30)
            
            # Volume indicators
            volume_ratio = data['Volume'] / data['Volume'].rolling(20).mean()
            
            # Volatility indicators
            volatility = data['Close'].rolling(20).std() / data['Close'].rolling(20).mean()
            
            # Trend strength
            trend_strength = (data['Close'] - data['Close'].rolling(50).mean()) / data['Close'].rolling(50).std()
            
            return {
                'price_momentum_1d': price_momentum_1d.iloc[-1] if len(price_momentum_1d) > 0 else 0,
                'price_momentum_7d': price_momentum_7d.iloc[-1] if len(price_momentum_7d) > 0 else 0,
                'price_momentum_30d': price_momentum_30d.iloc[-1] if len(price_momentum_30d) > 0 else 0,
                'volume_surge': volume_ratio.iloc[-1] if len(volume_ratio) > 0 else 1,
                'volatility_level': volatility.iloc[-1] if len(volatility) > 0 else 0,
                'trend_strength': trend_strength.iloc[-1] if len(trend_strength) > 0 else 0,
                'sentiment_score': self._calculate_sentiment_score(price_momentum_7d, volume_ratio, volatility)
            }
        except Exception as e:
            print(f"⚠️ Error calculating sentiment metrics: {str(e)}")
            return {}
    
    def _calculate_sentiment_score(self, price_momentum: pd.Series, volume_ratio: pd.Series, volatility: pd.Series) -> float:
        """Calculate overall sentiment score"""
        try:
            # Weighted sentiment calculation
            momentum_score = np.tanh(price_momentum.iloc[-1] * 10) * 50 + 50 if len(price_momentum) > 0 else 50
            volume_score = min(100, max(0, (volume_ratio.iloc[-1] - 1) * 50 + 50)) if len(volume_ratio) > 0 else 50
            volatility_score = max(0, 100 - volatility.iloc[-1] * 100) if len(volatility) > 0 else 50
            
            # Weighted average
            sentiment_score = (momentum_score * 0.5 + volume_score * 0.3 + volatility_score * 0.2)
            return max(0, min(100, sentiment_score))
        except Exception:
            return 50.0
    
    async def _train_price_model(self, data: pd.DataFrame):
        """Train price prediction model"""
        try:
            print("🤖 Training price prediction model...")
            
            # Prepare features
            feature_columns = [
                'sma_20', 'sma_50', 'rsi', 'macd', 'volatility', 'price_change_1d',
                'price_change_7d', 'price_change_30d', 'volume_ratio', 'market_cap',
                'pe_ratio', 'beta', 'sentiment_score'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['future_return_30d'].fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Create a separate scaler for price model
            from sklearn.preprocessing import MinMaxScaler
            price_scaler = MinMaxScaler()
            X_scaled = price_scaler.fit_transform(X)
            
            # Train model
            self.models['price_predictor'].fit(X_scaled, y)
            
            # Store feature names and scaler for later use
            self.trained_models['price_features'] = available_features
            self.trained_models['price_scaler'] = price_scaler
            
            print("✅ Price prediction model trained")
            
        except Exception as e:
            print(f"❌ Error training price model: {str(e)}")
    
    async def _train_trend_model(self, data: pd.DataFrame):
        """Train trend analysis model"""
        try:
            print("🤖 Training trend analysis model...")
            
            # Prepare features for trend prediction
            feature_columns = [
                'sma_20', 'sma_50', 'rsi', 'macd', 'volatility', 'price_change_1d',
                'price_change_7d', 'volume_ratio', 'sentiment_score', 'trend_strength'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            
            # Create trend target (1 for uptrend, 0 for downtrend)
            y = (data['future_return_7d'] > 0).astype(int)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Create a separate scaler for trend model
            from sklearn.preprocessing import MinMaxScaler
            trend_scaler = MinMaxScaler()
            X_scaled = trend_scaler.fit_transform(X)
            
            # Train model
            self.models['trend_analyzer'].fit(X_scaled, y)
            
            self.trained_models['trend_features'] = available_features
            self.trained_models['trend_scaler'] = trend_scaler
            
            print("✅ Trend analysis model trained")
            
        except Exception as e:
            print(f"❌ Error training trend model: {str(e)}")
    
    async def _train_risk_model(self, data: pd.DataFrame):
        """Train risk assessment model"""
        try:
            print("🤖 Training risk assessment model...")
            
            # Prepare features for risk prediction
            feature_columns = [
                'volatility', 'beta', 'debt_to_equity', 'current_ratio', 'price_change_30d',
                'volume_ratio', 'market_cap', 'pe_ratio', 'sentiment_score'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['future_volatility'].fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Create a separate scaler for risk model
            from sklearn.preprocessing import MinMaxScaler
            risk_scaler = MinMaxScaler()
            X_scaled = risk_scaler.fit_transform(X)
            
            # Train model
            self.models['risk_assessor'].fit(X_scaled, y)
            
            self.trained_models['risk_features'] = available_features
            self.trained_models['risk_scaler'] = risk_scaler
            
            print("✅ Risk assessment model trained")
            
        except Exception as e:
            print(f"❌ Error training risk model: {str(e)}")
    
    async def _train_sentiment_model(self, data: pd.DataFrame):
        """Train sentiment analysis model"""
        try:
            print("🤖 Training sentiment analysis model...")
            
            # Prepare features for sentiment prediction
            feature_columns = [
                'price_momentum_1d', 'price_momentum_7d', 'price_momentum_30d',
                'volume_surge', 'volatility_level', 'trend_strength'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            y = data['sentiment_score'].fillna(50)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 50)
            
            # Create a separate scaler for sentiment model
            from sklearn.preprocessing import MinMaxScaler
            sentiment_scaler = MinMaxScaler()
            X_scaled = sentiment_scaler.fit_transform(X)
            
            # Train model
            self.models['sentiment_analyzer'].fit(X_scaled, y)
            
            self.trained_models['sentiment_features'] = available_features
            self.trained_models['sentiment_scaler'] = sentiment_scaler
            
            print("✅ Sentiment analysis model trained")
            
        except Exception as e:
            print(f"❌ Error training sentiment model: {str(e)}")
    
    async def _train_fundamental_model(self, data: pd.DataFrame):
        """Train fundamental scoring model"""
        try:
            print("🤖 Training fundamental scoring model...")
            
            # Prepare features for fundamental analysis
            feature_columns = [
                'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book', 'price_to_sales',
                'debt_to_equity', 'current_ratio', 'gross_margin', 'operating_margin',
                'profit_margin', 'return_on_equity', 'return_on_assets', 'revenue_growth',
                'earnings_growth', 'dividend_yield', 'beta', 'free_cashflow', 'total_cash',
                'total_debt', 'book_value'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            
            # Create fundamental score target (based on future performance)
            y = data['future_return_30d'].fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Create a separate scaler for fundamental model
            from sklearn.preprocessing import MinMaxScaler
            fundamental_scaler = MinMaxScaler()
            X_scaled = fundamental_scaler.fit_transform(X)
            
            # Train model
            self.models['fundamental_scorer'].fit(X_scaled, y)
            
            self.trained_models['fundamental_features'] = available_features
            self.trained_models['fundamental_scaler'] = fundamental_scaler
            
            print("✅ Fundamental scoring model trained")
            
        except Exception as e:
            print(f"❌ Error training fundamental model: {str(e)}")
    
    async def analyze_company(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive company analysis using trained AI models"""
        try:
            print(f"🔍 Analyzing {symbol} with AI models...")
            
            # Get current data
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period="1y")
            info = ticker.info
            
            if hist_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate current metrics
            hist_data = self._calculate_technical_indicators(hist_data)
            fundamental_metrics = self._calculate_fundamental_metrics(info, hist_data)
            sentiment_metrics = self._calculate_sentiment_metrics(hist_data)
            
            # Prepare features for AI models
            current_features = self._prepare_current_features(
                hist_data, fundamental_metrics, sentiment_metrics
            )
            
            # Get AI predictions
            price_prediction = await self._predict_price(current_features)
            trend_prediction = await self._predict_trend(current_features)
            risk_assessment = await self._assess_risk(current_features)
            sentiment_analysis = await self._analyze_sentiment(current_features)
            fundamental_score = await self._score_fundamentals(current_features)
            
            # Calculate overall AI score
            overall_score = self._calculate_overall_ai_score(
                price_prediction, trend_prediction, risk_assessment, 
                sentiment_analysis, fundamental_score
            )
            
            return {
                'symbol': symbol,
                'analysis_date': datetime.now().isoformat(),
                'ai_score': overall_score,
                'price_prediction': price_prediction,
                'trend_analysis': trend_prediction,
                'risk_assessment': risk_assessment,
                'sentiment_analysis': sentiment_analysis,
                'fundamental_score': fundamental_score,
                'recommendation': self._generate_ai_recommendation(overall_score),
                'confidence': self._calculate_confidence(price_prediction, trend_prediction, risk_assessment),
                'key_insights': self._generate_key_insights(
                    symbol, price_prediction, trend_prediction, risk_assessment, sentiment_analysis
                )
            }
            
        except Exception as e:
            raise Exception(f"Error in AI company analysis: {str(e)}")
    
    def _prepare_current_features(self, hist_data: pd.DataFrame, 
                                fundamental_metrics: Dict, sentiment_metrics: Dict) -> Dict:
        """Prepare current features for AI models"""
        try:
            current_price = hist_data['Close'].iloc[-1]
            
            features = {
                'sma_20': hist_data['SMA_20'].iloc[-1] if 'SMA_20' in hist_data.columns else current_price,
                'sma_50': hist_data['SMA_50'].iloc[-1] if 'SMA_50' in hist_data.columns else current_price,
                'rsi': hist_data['RSI'].iloc[-1] if 'RSI' in hist_data.columns else 50,
                'macd': hist_data['MACD'].iloc[-1] if 'MACD' in hist_data.columns else 0,
                'volatility': hist_data['Close'].iloc[-20:].std() if len(hist_data) >= 20 else 0,
                'price_change_1d': hist_data['Close'].pct_change().iloc[-1] if len(hist_data) > 1 else 0,
                'price_change_7d': hist_data['Close'].pct_change(7).iloc[-1] if len(hist_data) > 7 else 0,
                'price_change_30d': hist_data['Close'].pct_change(30).iloc[-1] if len(hist_data) > 30 else 0,
                'volume_ratio': hist_data['Volume'].iloc[-1] / hist_data['Volume'].iloc[-20:].mean() if len(hist_data) >= 20 else 1,
                'current_price': current_price,
                **fundamental_metrics,
                **sentiment_metrics
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ Error preparing features: {str(e)}")
            return {}
    
    async def _predict_price(self, features: Dict) -> Dict[str, Any]:
        """Predict future price using trained AI model"""
        try:
            if 'price_features' not in self.trained_models or 'price_scaler' not in self.trained_models:
                return {'prediction': 0, 'confidence': 0.5, 'direction': 'neutral'}
            
            # Prepare features for model
            model_features = [features.get(feat, 0) for feat in self.trained_models['price_features']]
            X = np.array(model_features).reshape(1, -1)
            
            # Scale features using the correct scaler
            X_scaled = self.trained_models['price_scaler'].transform(X)
            
            # Make prediction
            predicted_return = self.models['price_predictor'].predict(X_scaled)[0]
            current_price = features.get('current_price', 100)
            predicted_price = current_price * (1 + predicted_return)
            
            # Calculate confidence based on model certainty
            confidence = min(0.95, max(0.1, abs(predicted_return) * 10 + 0.5))
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_return': predicted_return,
                'confidence': confidence,
                'direction': 'bullish' if predicted_return > 0.02 else 'bearish' if predicted_return < -0.02 else 'neutral'
            }
            
        except Exception as e:
            print(f"⚠️ Error in price prediction: {str(e)}")
            return {'prediction': 0, 'confidence': 0.5, 'direction': 'neutral'}
    
    async def _predict_trend(self, features: Dict) -> Dict[str, Any]:
        """Predict trend direction using trained AI model"""
        try:
            if 'trend_features' not in self.trained_models or 'trend_scaler' not in self.trained_models:
                return {'direction': 'neutral', 'strength': 0.5, 'confidence': 0.5}
            
            # Prepare features for model
            model_features = [features.get(feat, 0) for feat in self.trained_models['trend_features']]
            X = np.array(model_features).reshape(1, -1)
            
            # Scale features using the correct scaler
            X_scaled = self.trained_models['trend_scaler'].transform(X)
            
            # Make prediction
            trend_probability = self.models['trend_analyzer'].predict(X_scaled)[0]
            
            # Determine direction and strength
            if trend_probability > 0.7:
                direction = 'strong_uptrend'
                strength = trend_probability
            elif trend_probability > 0.6:
                direction = 'uptrend'
                strength = trend_probability
            elif trend_probability < 0.3:
                direction = 'strong_downtrend'
                strength = 1 - trend_probability
            elif trend_probability < 0.4:
                direction = 'downtrend'
                strength = 1 - trend_probability
            else:
                direction = 'sideways'
                strength = 0.5
            
            return {
                'direction': direction,
                'strength': strength,
                'probability': trend_probability,
                'confidence': min(0.95, max(0.1, abs(trend_probability - 0.5) * 2 + 0.5))
            }
            
        except Exception as e:
            print(f"⚠️ Error in trend prediction: {str(e)}")
            return {'direction': 'neutral', 'strength': 0.5, 'confidence': 0.5}
    
    async def _assess_risk(self, features: Dict) -> Dict[str, Any]:
        """Assess risk using trained AI model"""
        try:
            if 'risk_features' not in self.trained_models or 'risk_scaler' not in self.trained_models:
                return {'risk_level': 'medium', 'score': 50, 'volatility': 0.2}
            
            # Prepare features for model
            model_features = [features.get(feat, 0) for feat in self.trained_models['risk_features']]
            X = np.array(model_features).reshape(1, -1)
            
            # Scale features using the correct scaler
            X_scaled = self.trained_models['risk_scaler'].transform(X)
            
            # Make prediction
            predicted_volatility = self.models['risk_assessor'].predict(X_scaled)[0]
            
            # Calculate risk score
            risk_score = min(100, max(0, predicted_volatility * 100))
            
            # Determine risk level
            if risk_score > 70:
                risk_level = 'very_high'
            elif risk_score > 50:
                risk_level = 'high'
            elif risk_score > 30:
                risk_level = 'medium'
            elif risk_score > 15:
                risk_level = 'low'
            else:
                risk_level = 'very_low'
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'predicted_volatility': predicted_volatility,
                'confidence': min(0.95, max(0.1, abs(predicted_volatility - 0.2) * 5 + 0.5))
            }
            
        except Exception as e:
            print(f"⚠️ Error in risk assessment: {str(e)}")
            return {'risk_level': 'medium', 'score': 50, 'volatility': 0.2}
    
    async def _analyze_sentiment(self, features: Dict) -> Dict[str, Any]:
        """Analyze sentiment using trained AI model"""
        try:
            if 'sentiment_features' not in self.trained_models or 'sentiment_scaler' not in self.trained_models:
                return {'sentiment': 'neutral', 'score': 50, 'confidence': 0.5}
            
            # Prepare features for model
            model_features = [features.get(feat, 0) for feat in self.trained_models['sentiment_features']]
            X = np.array(model_features).reshape(1, -1)
            
            # Scale features using the correct scaler
            X_scaled = self.trained_models['sentiment_scaler'].transform(X)
            
            # Make prediction
            sentiment_score = self.models['sentiment_analyzer'].predict(X_scaled)[0]
            sentiment_score = max(0, min(100, sentiment_score))
            
            # Determine sentiment
            if sentiment_score > 70:
                sentiment = 'very_positive'
            elif sentiment_score > 60:
                sentiment = 'positive'
            elif sentiment_score > 40:
                sentiment = 'neutral'
            elif sentiment_score > 30:
                sentiment = 'negative'
            else:
                sentiment = 'very_negative'
            
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': min(0.95, max(0.1, abs(sentiment_score - 50) / 50 + 0.5))
            }
            
        except Exception as e:
            print(f"⚠️ Error in sentiment analysis: {str(e)}")
            return {'sentiment': 'neutral', 'score': 50, 'confidence': 0.5}
    
    async def _score_fundamentals(self, features: Dict) -> Dict[str, Any]:
        """Score fundamentals using trained AI model"""
        try:
            if 'fundamental_features' not in self.trained_models or 'fundamental_scaler' not in self.trained_models:
                return {'score': 50, 'grade': 'C', 'confidence': 0.5}
            
            # Prepare features for model
            model_features = [features.get(feat, 0) for feat in self.trained_models['fundamental_features']]
            X = np.array(model_features).reshape(1, -1)
            
            # Scale features using the correct scaler
            X_scaled = self.trained_models['fundamental_scaler'].transform(X)
            
            # Make prediction
            fundamental_return = self.models['fundamental_scorer'].predict(X_scaled)[0]
            
            # Convert to score (0-100)
            fundamental_score = min(100, max(0, (fundamental_return + 0.2) * 250))
            
            # Determine grade
            if fundamental_score > 80:
                grade = 'A+'
            elif fundamental_score > 70:
                grade = 'A'
            elif fundamental_score > 60:
                grade = 'B+'
            elif fundamental_score > 50:
                grade = 'B'
            elif fundamental_score > 40:
                grade = 'C+'
            elif fundamental_score > 30:
                grade = 'C'
            elif fundamental_score > 20:
                grade = 'D'
            else:
                grade = 'F'
            
            return {
                'score': fundamental_score,
                'grade': grade,
                'expected_return': fundamental_return,
                'confidence': min(0.95, max(0.1, abs(fundamental_return) * 5 + 0.5))
            }
            
        except Exception as e:
            print(f"⚠️ Error in fundamental scoring: {str(e)}")
            return {'score': 50, 'grade': 'C', 'confidence': 0.5}
    
    def _calculate_overall_ai_score(self, price_prediction: Dict, trend_prediction: Dict, 
                                  risk_assessment: Dict, sentiment_analysis: Dict, 
                                  fundamental_score: Dict) -> float:
        """Calculate overall AI score"""
        try:
            # Weighted scoring
            weights = {
                'price': 0.25,
                'trend': 0.20,
                'risk': 0.20,
                'sentiment': 0.15,
                'fundamental': 0.20
            }
            
            # Calculate component scores
            price_score = 50 + (price_prediction.get('expected_return', 0) * 1000)
            trend_score = trend_prediction.get('strength', 0.5) * 100
            risk_score = 100 - risk_assessment.get('risk_score', 50)  # Invert risk
            sentiment_score = sentiment_analysis.get('score', 50)
            fundamental_score_val = fundamental_score.get('score', 50)
            
            # Calculate weighted average
            overall_score = (
                price_score * weights['price'] +
                trend_score * weights['trend'] +
                risk_score * weights['risk'] +
                sentiment_score * weights['sentiment'] +
                fundamental_score_val * weights['fundamental']
            )
            
            return max(0, min(100, overall_score))
            
        except Exception as e:
            print(f"⚠️ Error calculating overall score: {str(e)}")
            return 50.0
    
    def _generate_ai_recommendation(self, overall_score: float) -> str:
        """Generate AI recommendation based on overall score"""
        if overall_score >= 85:
            return "Strong Buy"
        elif overall_score >= 70:
            return "Buy"
        elif overall_score >= 55:
            return "Hold"
        elif overall_score >= 40:
            return "Sell"
        else:
            return "Strong Sell"
    
    def _calculate_confidence(self, price_prediction: Dict, trend_prediction: Dict, 
                            risk_assessment: Dict) -> float:
        """Calculate overall confidence in analysis"""
        try:
            confidences = [
                price_prediction.get('confidence', 0.5),
                trend_prediction.get('confidence', 0.5),
                risk_assessment.get('confidence', 0.5)
            ]
            
            return sum(confidences) / len(confidences)
            
        except Exception:
            return 0.5
    
    def _generate_key_insights(self, symbol: str, price_prediction: Dict, 
                             trend_prediction: Dict, risk_assessment: Dict, 
                             sentiment_analysis: Dict) -> List[str]:
        """Generate key insights from AI analysis"""
        insights = []
        
        try:
            # Price insights
            if price_prediction.get('direction') == 'bullish':
                insights.append(f"AI predicts bullish price movement for {symbol}")
            elif price_prediction.get('direction') == 'bearish':
                insights.append(f"AI predicts bearish price movement for {symbol}")
            
            # Trend insights
            if trend_prediction.get('direction') in ['strong_uptrend', 'uptrend']:
                insights.append("Strong upward trend detected by AI models")
            elif trend_prediction.get('direction') in ['strong_downtrend', 'downtrend']:
                insights.append("Downward trend detected by AI models")
            
            # Risk insights
            if risk_assessment.get('risk_level') in ['very_high', 'high']:
                insights.append("High risk detected - consider position sizing")
            elif risk_assessment.get('risk_level') in ['very_low', 'low']:
                insights.append("Low risk profile - suitable for conservative investors")
            
            # Sentiment insights
            if sentiment_analysis.get('sentiment') in ['very_positive', 'positive']:
                insights.append("Positive market sentiment detected")
            elif sentiment_analysis.get('sentiment') in ['very_negative', 'negative']:
                insights.append("Negative market sentiment detected")
            
            # Add general insights
            insights.append("Analysis based on comprehensive AI models trained on market data")
            insights.append("Recommendations should be combined with your own research")
            
        except Exception as e:
            insights.append("AI analysis completed with standard insights")
        
        return insights[:5]  # Return top 5 insights
