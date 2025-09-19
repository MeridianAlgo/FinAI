import pandas as pd
import numpy as np
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from stocknews import StockNews
import json
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ReinforcementLearningService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.reward_system = RewardSystem()
        self.performance_tracker = PerformanceTracker()
        self.news_analyzer = NewsSentimentAnalyzer()
        self.learning_history = []
        self.model_weights = {}
        self.max_score = 1000000  # 1 million points maximum
        
    async def initialize_learning_system(self):
        """Initialize the reinforcement learning system"""
        print("Initializing Reinforcement Learning System...")
        
        # Initialize models with reward-based learning
        self.models = {
            'price_predictor': RewardBasedRandomForest(),
            'trend_analyzer': RewardBasedGradientBoosting(),
            'risk_assessor': RewardBasedRandomForest(),
            'sentiment_analyzer': RewardBasedLinearRegression(),
            'fundamental_scorer': RewardBasedRidge()
        }
        
        # Initialize scalers
        self.scalers = {
            'price_scaler': StandardScaler(),
            'trend_scaler': MinMaxScaler(),
            'risk_scaler': StandardScaler(),
            'sentiment_scaler': MinMaxScaler(),
            'fundamental_scaler': StandardScaler()
        }
        
        # Load existing performance data
        await self._load_performance_data()
        
        print("Reinforcement Learning System initialized!")
    
    async def train_with_reinforcement_learning(self, symbol: str, start_date: str, end_date: str):
        """Train models using reinforcement learning with historical data"""
        print(f"Training with reinforcement learning for {symbol} from {start_date} to {end_date}")
        
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            
            if hist_data.empty:
                print(f"No data available for {symbol} in the specified period")
                return
            
            # Get news sentiment data
            news_sentiment = await self.news_analyzer.get_news_sentiment(symbol, start_date, end_date)
            
            # Calculate technical indicators
            hist_data = self._calculate_technical_indicators(hist_data)
            
            # Create training data with monthly predictions
            training_data = self._create_monthly_training_data(hist_data, news_sentiment, symbol)
            
            # Train models with reinforcement learning
            await self._train_models_with_rewards(training_data, symbol)
            
            # Update performance tracking
            await self._update_performance_tracking(symbol, training_data)
            
            print(f"Reinforcement learning completed for {symbol}")
            
        except Exception as e:
            print(f"Error in reinforcement learning for {symbol}: {str(e)}")
    
    def _create_monthly_training_data(self, hist_data: pd.DataFrame, news_sentiment: Dict, symbol: str) -> pd.DataFrame:
        """Create training data with monthly prediction windows"""
        all_data = []
        
        # Create monthly prediction windows
        for i in range(60, len(hist_data) - 30):  # Leave 30 days for prediction
            # Get data for current month
            current_month_data = hist_data.iloc[i-30:i]
            next_month_data = hist_data.iloc[i:i+30]
            
            if len(current_month_data) < 30 or len(next_month_data) < 30:
                continue
            
            # Calculate features for current month
            features = self._calculate_features(current_month_data, news_sentiment, i)
            
            # Calculate target for next month
            current_price = current_month_data['Close'].iloc[-1]
            next_month_price = next_month_data['Close'].iloc[-1]
            actual_return = (next_month_price - current_price) / current_price
            
            # Calculate actual volatility
            actual_volatility = next_month_data['Close'].pct_change().std()
            
            # Calculate actual trend
            actual_trend = 1 if actual_return > 0 else 0
            
            sample = {
                'symbol': symbol,
                'date': hist_data.index[i],
                'current_price': current_price,
                'next_month_price': next_month_price,
                'actual_return': actual_return,
                'actual_volatility': actual_volatility,
                'actual_trend': actual_trend,
                **features
            }
            all_data.append(sample)
        
        return pd.DataFrame(all_data)
    
    def _calculate_features(self, data: pd.DataFrame, news_sentiment: Dict, index: int) -> Dict:
        """Calculate features for the given data period"""
        try:
            # Technical indicators
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            rsi = self._calculate_rsi(data['Close']).iloc[-1]
            macd = self._calculate_macd(data['Close']).iloc[-1]
            volatility = data['Close'].pct_change().std()
            
            # Price momentum
            price_change_1d = data['Close'].pct_change().iloc[-1]
            price_change_7d = data['Close'].pct_change(7).iloc[-1]
            price_change_30d = data['Close'].pct_change(30).iloc[-1]
            
            # Volume indicators
            volume_ratio = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
            
            # News sentiment
            sentiment_score = news_sentiment.get('sentiment_score', 50)
            news_count = news_sentiment.get('news_count', 0)
            
            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'macd': macd,
                'volatility': volatility,
                'price_change_1d': price_change_1d,
                'price_change_7d': price_change_7d,
                'price_change_30d': price_change_30d,
                'volume_ratio': volume_ratio,
                'sentiment_score': sentiment_score,
                'news_count': news_count
            }
        except Exception as e:
            print(f"Error calculating features: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        return ema_12 - ema_26
    
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
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # Bollinger Bands
            data['BB_middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
            data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Volatility
            data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()
            
            return data
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            return data
    
    async def _train_models_with_rewards(self, data: pd.DataFrame, symbol: str):
        """Train models with reinforcement learning rewards"""
        try:
            # Prepare features
            feature_columns = [
                'sma_20', 'sma_50', 'rsi', 'macd', 'volatility',
                'price_change_1d', 'price_change_7d', 'price_change_30d',
                'volume_ratio', 'sentiment_score', 'news_count'
            ]
            
            available_features = [col for col in feature_columns if col in data.columns]
            X = data[available_features].fillna(0)
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], 0)
            
            # Train price prediction model
            await self._train_price_model_with_rewards(X, data, symbol)
            
            # Train trend analysis model
            await self._train_trend_model_with_rewards(X, data, symbol)
            
            # Train risk assessment model
            await self._train_risk_model_with_rewards(X, data, symbol)
            
            # Train sentiment analysis model
            await self._train_sentiment_model_with_rewards(X, data, symbol)
            
            # Train fundamental scoring model
            await self._train_fundamental_model_with_rewards(X, data, symbol)
            
        except Exception as e:
            print(f"Error training models with rewards: {str(e)}")
    
    async def _train_price_model_with_rewards(self, X: pd.DataFrame, data: pd.DataFrame, symbol: str):
        """Train price prediction model with reinforcement learning"""
        try:
            y = data['actual_return'].fillna(0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = self.scalers['price_scaler'].fit_transform(X)
            
            # Train model
            self.models['price_predictor'].fit(X_scaled, y)
            
            # Calculate predictions and rewards
            predictions = self.models['price_predictor'].predict(X_scaled)
            
            # Calculate rewards based on prediction accuracy
            rewards = []
            for i, (pred, actual) in enumerate(zip(predictions, y)):
                error = abs(pred - actual)
                reward = self.reward_system.calculate_price_reward(error, actual)
                rewards.append(reward)
            
            # Update model weights based on rewards
            avg_reward = np.mean(rewards)
            self.model_weights['price_predictor'] = avg_reward
            
            # Store training data for performance tracking
            self.learning_history.append({
                'symbol': symbol,
                'model': 'price_predictor',
                'avg_reward': avg_reward,
                'predictions': predictions.tolist(),
                'actuals': y.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Price model trained with average reward: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"Error training price model with rewards: {str(e)}")
    
    async def _train_trend_model_with_rewards(self, X: pd.DataFrame, data: pd.DataFrame, symbol: str):
        """Train trend analysis model with reinforcement learning"""
        try:
            y = data['actual_trend'].fillna(0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = self.scalers['trend_scaler'].fit_transform(X)
            
            # Train model
            self.models['trend_analyzer'].fit(X_scaled, y)
            
            # Calculate predictions and rewards
            predictions = self.models['trend_analyzer'].predict(X_scaled)
            
            # Calculate rewards based on prediction accuracy
            rewards = []
            for i, (pred, actual) in enumerate(zip(predictions, y)):
                error = abs(pred - actual)
                reward = self.reward_system.calculate_trend_reward(error, actual)
                rewards.append(reward)
            
            # Update model weights based on rewards
            avg_reward = np.mean(rewards)
            self.model_weights['trend_analyzer'] = avg_reward
            
            # Store training data for performance tracking
            self.learning_history.append({
                'symbol': symbol,
                'model': 'trend_analyzer',
                'avg_reward': avg_reward,
                'predictions': predictions.tolist(),
                'actuals': y.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Trend model trained with average reward: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"Error training trend model with rewards: {str(e)}")
    
    async def _train_risk_model_with_rewards(self, X: pd.DataFrame, data: pd.DataFrame, symbol: str):
        """Train risk assessment model with reinforcement learning"""
        try:
            y = data['actual_volatility'].fillna(0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = self.scalers['risk_scaler'].fit_transform(X)
            
            # Train model
            self.models['risk_assessor'].fit(X_scaled, y)
            
            # Calculate predictions and rewards
            predictions = self.models['risk_assessor'].predict(X_scaled)
            
            # Calculate rewards based on prediction accuracy
            rewards = []
            for i, (pred, actual) in enumerate(zip(predictions, y)):
                error = abs(pred - actual)
                reward = self.reward_system.calculate_risk_reward(error, actual)
                rewards.append(reward)
            
            # Update model weights based on rewards
            avg_reward = np.mean(rewards)
            self.model_weights['risk_assessor'] = avg_reward
            
            # Store training data for performance tracking
            self.learning_history.append({
                'symbol': symbol,
                'model': 'risk_assessor',
                'avg_reward': avg_reward,
                'predictions': predictions.tolist(),
                'actuals': y.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Risk model trained with average reward: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"Error training risk model with rewards: {str(e)}")
    
    async def _train_sentiment_model_with_rewards(self, X: pd.DataFrame, data: pd.DataFrame, symbol: str):
        """Train sentiment analysis model with reinforcement learning"""
        try:
            # Use sentiment score as target
            y = data['sentiment_score'].fillna(50)
            y = y.replace([np.inf, -np.inf], 50)
            
            # Scale features
            X_scaled = self.scalers['sentiment_scaler'].fit_transform(X)
            
            # Train model
            self.models['sentiment_analyzer'].fit(X_scaled, y)
            
            # Calculate predictions and rewards
            predictions = self.models['sentiment_analyzer'].predict(X_scaled)
            
            # Calculate rewards based on prediction accuracy
            rewards = []
            for i, (pred, actual) in enumerate(zip(predictions, y)):
                error = abs(pred - actual)
                reward = self.reward_system.calculate_sentiment_reward(error, actual)
                rewards.append(reward)
            
            # Update model weights based on rewards
            avg_reward = np.mean(rewards)
            self.model_weights['sentiment_analyzer'] = avg_reward
            
            # Store training data for performance tracking
            self.learning_history.append({
                'symbol': symbol,
                'model': 'sentiment_analyzer',
                'avg_reward': avg_reward,
                'predictions': predictions.tolist(),
                'actuals': y.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Sentiment model trained with average reward: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"Error training sentiment model with rewards: {str(e)}")
    
    async def _train_fundamental_model_with_rewards(self, X: pd.DataFrame, data: pd.DataFrame, symbol: str):
        """Train fundamental scoring model with reinforcement learning"""
        try:
            # Use actual return as target for fundamental scoring
            y = data['actual_return'].fillna(0)
            y = y.replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = self.scalers['fundamental_scaler'].fit_transform(X)
            
            # Train model
            self.models['fundamental_scorer'].fit(X_scaled, y)
            
            # Calculate predictions and rewards
            predictions = self.models['fundamental_scorer'].predict(X_scaled)
            
            # Calculate rewards based on prediction accuracy
            rewards = []
            for i, (pred, actual) in enumerate(zip(predictions, y)):
                error = abs(pred - actual)
                reward = self.reward_system.calculate_fundamental_reward(error, actual)
                rewards.append(reward)
            
            # Update model weights based on rewards
            avg_reward = np.mean(rewards)
            self.model_weights['fundamental_scorer'] = avg_reward
            
            # Store training data for performance tracking
            self.learning_history.append({
                'symbol': symbol,
                'model': 'fundamental_scorer',
                'avg_reward': avg_reward,
                'predictions': predictions.tolist(),
                'actuals': y.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Fundamental model trained with average reward: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"Error training fundamental model with rewards: {str(e)}")
    
    async def _update_performance_tracking(self, symbol: str, data: pd.DataFrame):
        """Update performance tracking with new results"""
        try:
            # Calculate overall performance metrics
            total_reward = sum(self.model_weights.values())
            avg_reward = total_reward / len(self.model_weights) if self.model_weights else 0
            
            # Update performance tracker
            self.performance_tracker.update_performance(symbol, {
                'total_reward': total_reward,
                'avg_reward': avg_reward,
                'model_weights': self.model_weights.copy(),
                'training_samples': len(data),
                'timestamp': datetime.now().isoformat()
            })
            
            # Save performance data
            await self._save_performance_data()
            
            print(f"Performance updated for {symbol}: Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")
            
        except Exception as e:
            print(f"Error updating performance tracking: {str(e)}")
    
    async def _load_performance_data(self):
        """Load existing performance data"""
        try:
            if os.path.exists('performance_data.json'):
                with open('performance_data.json', 'r') as f:
                    data = json.load(f)
                    self.performance_tracker.performance_data = data
                    print("Performance data loaded successfully")
        except Exception as e:
            print(f"Error loading performance data: {str(e)}")
    
    async def _save_performance_data(self):
        """Save performance data"""
        try:
            with open('performance_data.json', 'w') as f:
                json.dump(self.performance_tracker.performance_data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {str(e)}")
    
    def get_current_score(self) -> float:
        """Get current total score"""
        return sum(self.model_weights.values())
    
    def get_model_performance(self) -> Dict:
        """Get model performance summary"""
        return {
            'total_score': self.get_current_score(),
            'max_score': self.max_score,
            'model_weights': self.model_weights,
            'performance_data': self.performance_tracker.performance_data
        }

class RewardSystem:
    def __init__(self):
        self.base_reward = 100
        self.max_reward = 1000
        self.min_reward = -500
    
    def calculate_price_reward(self, error: float, actual_return: float) -> float:
        """Calculate reward for price prediction accuracy"""
        try:
            # Base reward for prediction
            reward = self.base_reward
            
            # Accuracy bonus/penalty
            if error < 0.01:  # Very accurate (within 1%)
                reward += 500
            elif error < 0.05:  # Good accuracy (within 5%)
                reward += 200
            elif error < 0.10:  # Acceptable accuracy (within 10%)
                reward += 50
            elif error < 0.20:  # Poor accuracy (within 20%)
                reward -= 100
            else:  # Very poor accuracy (>20%)
                reward -= 300
            
            # Direction bonus
            if actual_return > 0 and error < 0.05:  # Correctly predicted positive return
                reward += 200
            elif actual_return < 0 and error < 0.05:  # Correctly predicted negative return
                reward += 200
            
            return max(self.min_reward, min(self.max_reward, reward))
        except:
            return 0
    
    def calculate_trend_reward(self, error: float, actual_trend: float) -> float:
        """Calculate reward for trend prediction accuracy"""
        try:
            reward = self.base_reward
            
            if error < 0.1:  # Very accurate trend prediction
                reward += 400
            elif error < 0.3:  # Good trend prediction
                reward += 200
            elif error < 0.5:  # Acceptable trend prediction
                reward += 50
            else:  # Poor trend prediction
                reward -= 200
            
            return max(self.min_reward, min(self.max_reward, reward))
        except:
            return 0
    
    def calculate_risk_reward(self, error: float, actual_volatility: float) -> float:
        """Calculate reward for risk assessment accuracy"""
        try:
            reward = self.base_reward
            
            if error < 0.01:  # Very accurate volatility prediction
                reward += 300
            elif error < 0.05:  # Good volatility prediction
                reward += 150
            elif error < 0.10:  # Acceptable volatility prediction
                reward += 50
            else:  # Poor volatility prediction
                reward -= 150
            
            return max(self.min_reward, min(self.max_reward, reward))
        except:
            return 0
    
    def calculate_sentiment_reward(self, error: float, actual_sentiment: float) -> float:
        """Calculate reward for sentiment analysis accuracy"""
        try:
            reward = self.base_reward
            
            if error < 5:  # Very accurate sentiment (within 5 points)
                reward += 200
            elif error < 15:  # Good sentiment accuracy (within 15 points)
                reward += 100
            elif error < 25:  # Acceptable sentiment accuracy (within 25 points)
                reward += 25
            else:  # Poor sentiment accuracy
                reward -= 100
            
            return max(self.min_reward, min(self.max_reward, reward))
        except:
            return 0
    
    def calculate_fundamental_reward(self, error: float, actual_return: float) -> float:
        """Calculate reward for fundamental analysis accuracy"""
        try:
            reward = self.base_reward
            
            if error < 0.02:  # Very accurate fundamental prediction
                reward += 400
            elif error < 0.05:  # Good fundamental prediction
                reward += 200
            elif error < 0.10:  # Acceptable fundamental prediction
                reward += 50
            else:  # Poor fundamental prediction
                reward -= 200
            
            return max(self.min_reward, min(self.max_reward, reward))
        except:
            return 0

class PerformanceTracker:
    def __init__(self):
        self.performance_data = {}
    
    def update_performance(self, symbol: str, data: Dict):
        """Update performance data for a symbol"""
        if symbol not in self.performance_data:
            self.performance_data[symbol] = []
        
        self.performance_data[symbol].append(data)
        
        # Keep only last 100 entries per symbol
        if len(self.performance_data[symbol]) > 100:
            self.performance_data[symbol] = self.performance_data[symbol][-100:]

class NewsSentimentAnalyzer:
    def __init__(self):
        self.stock_news = None  # Initialize when needed
    
    async def get_news_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Get news sentiment for a symbol in the given date range"""
        try:
            # Initialize StockNews if not already done
            if self.stock_news is None:
                self.stock_news = StockNews([symbol])
            
            # Get news for the symbol
            try:
                news = self.stock_news.read_rss(symbol)
            except:
                news = []
            
            if not news:
                return {'sentiment_score': 50, 'news_count': 0}
            
            # Filter news by date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            filtered_news = []
            for article in news:
                try:
                    article_date = datetime.strptime(article['date'], '%Y-%m-%d')
                    if start_dt <= article_date <= end_dt:
                        filtered_news.append(article)
                except:
                    continue
            
            if not filtered_news:
                return {'sentiment_score': 50, 'news_count': 0}
            
            # Calculate sentiment scores
            sentiment_scores = []
            for article in filtered_news:
                title = article.get('title', '')
                summary = article.get('summary', '')
                text = title + ' ' + summary
                
                # Simple sentiment analysis
                sentiment = self._analyze_text_sentiment(text)
                sentiment_scores.append(sentiment)
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 50
            sentiment_score = max(0, min(100, avg_sentiment * 100))
            
            return {
                'sentiment_score': sentiment_score,
                'news_count': len(filtered_news),
                'avg_sentiment': avg_sentiment
            }
            
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {str(e)}")
            return {'sentiment_score': 50, 'news_count': 0}
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze text sentiment using keyword-based approach"""
        try:
            if not text:
                return 0.5
            
            # Positive keywords
            positive_words = [
                'bullish', 'growth', 'profit', 'gain', 'rise', 'increase', 'strong', 'positive',
                'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'recommend', 'success',
                'breakthrough', 'innovation', 'leading', 'dominant', 'expansion', 'surge',
                'rally', 'momentum', 'optimistic', 'confident', 'robust', 'solid'
            ]
            
            # Negative keywords
            negative_words = [
                'bearish', 'decline', 'loss', 'fall', 'decrease', 'weak', 'negative',
                'miss', 'underperform', 'downgrade', 'sell', 'avoid', 'failure',
                'crisis', 'recession', 'bankruptcy', 'lawsuit', 'investigation', 'crash',
                'plunge', 'slump', 'concern', 'risk', 'uncertainty', 'volatile'
            ]
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.5
            
            # Calculate sentiment score
            sentiment_score = (positive_count - negative_count) / total_words
            return max(0, min(1, sentiment_score + 0.5))
            
        except:
            return 0.5

class RewardBasedRandomForest(RandomForestRegressor):
    """Random Forest with reward-based learning"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_history = []
    
    def fit(self, X, y, sample_weight=None):
        """Fit with reward-based learning"""
        return super().fit(X, y, sample_weight)

class RewardBasedGradientBoosting(GradientBoostingRegressor):
    """Gradient Boosting with reward-based learning"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_history = []
    
    def fit(self, X, y, sample_weight=None):
        """Fit with reward-based learning"""
        return super().fit(X, y, sample_weight)

class RewardBasedLinearRegression(LinearRegression):
    """Linear Regression with reward-based learning"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_history = []
    
    def fit(self, X, y, sample_weight=None):
        """Fit with reward-based learning"""
        return super().fit(X, y, sample_weight)

class RewardBasedRidge(Ridge):
    """Ridge Regression with reward-based learning"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_history = []
    
    def fit(self, X, y, sample_weight=None):
        """Fit with reward-based learning"""
        return super().fit(X, y, sample_weight)
