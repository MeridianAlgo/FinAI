import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class PredictionService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    async def predict_price(self, symbol: str, timeframe: str = "1y", prediction_days: int = 30, model_type: str = "lstm") -> Dict[str, Any]:
        """Predict future stock prices using AI models"""
        try:
            # Get historical data
            data = await self._get_historical_data(symbol, timeframe)
            
            if data.empty or len(data) < 50:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Prepare features
            features = self._prepare_features(data)
            
            # Train model and make predictions
            if model_type == "lstm":
                predictions = await self._lstm_prediction(features, prediction_days)
            elif model_type == "arima":
                predictions = await self._arima_prediction(data, prediction_days)
            elif model_type == "linear_regression":
                predictions = await self._linear_regression_prediction(features, prediction_days)
            else:
                predictions = await self._ensemble_prediction(features, data, prediction_days)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(predictions, data)
            
            # Generate prediction dates
            last_date = data.index[-1]
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
            
            return {
                "symbol": symbol,
                "model_type": model_type,
                "prediction_days": prediction_days,
                "predictions": [
                    {
                        "date": date.isoformat(),
                        "predicted_price": float(pred),
                        "confidence_lower": float(ci_lower),
                        "confidence_upper": float(ci_upper)
                    }
                    for date, pred, ci_lower, ci_upper in zip(
                        prediction_dates, 
                        predictions, 
                        confidence_intervals['lower'], 
                        confidence_intervals['upper']
                    )
                ],
                "current_price": float(data['Close'].iloc[-1]),
                "prediction_accuracy": self._calculate_model_accuracy(features, data),
                "trend_direction": self._determine_trend_direction(predictions),
                "risk_assessment": self._assess_prediction_risk(predictions, data)
            }
        except Exception as e:
            raise Exception(f"Error in price prediction: {str(e)}")
    
    async def predict_trend(self, symbol: str, timeframe: str = "1y", prediction_days: int = 30) -> Dict[str, Any]:
        """Predict market trend direction"""
        try:
            data = await self._get_historical_data(symbol, timeframe)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate technical indicators
            data = self._add_technical_indicators(data)
            
            # Prepare trend features
            trend_features = self._prepare_trend_features(data)
            
            # Predict trend using ensemble method
            trend_predictions = await self._predict_trend_direction(trend_features, prediction_days)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(trend_predictions, data)
            
            return {
                "symbol": symbol,
                "prediction_days": prediction_days,
                "trend_direction": trend_predictions['direction'],
                "trend_strength": trend_strength,
                "confidence": trend_predictions['confidence'],
                "key_levels": self._identify_key_levels(data),
                "support_resistance": self._find_support_resistance_levels(data),
                "volatility_forecast": self._forecast_volatility(data, prediction_days)
            }
        except Exception as e:
            raise Exception(f"Error in trend prediction: {str(e)}")
    
    async def predict_volatility(self, symbol: str, timeframe: str = "1y") -> Dict[str, Any]:
        """Predict future volatility"""
        try:
            data = await self._get_historical_data(symbol, timeframe)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Calculate returns and volatility
            returns = data['Close'].pct_change().dropna()
            current_volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Predict future volatility using GARCH-like approach
            volatility_forecast = self._forecast_volatility_garch(returns)
            
            # Calculate volatility percentiles
            volatility_percentiles = self._calculate_volatility_percentiles(returns)
            
            return {
                "symbol": symbol,
                "current_volatility": current_volatility,
                "predicted_volatility": volatility_forecast['predicted'],
                "volatility_trend": volatility_forecast['trend'],
                "volatility_percentiles": volatility_percentiles,
                "risk_level": self._assess_volatility_risk(volatility_forecast['predicted']),
                "historical_volatility": {
                    "1m": returns.tail(21).std() * np.sqrt(252),
                    "3m": returns.tail(63).std() * np.sqrt(252),
                    "6m": returns.tail(126).std() * np.sqrt(252),
                    "1y": returns.std() * np.sqrt(252)
                }
            }
        except Exception as e:
            raise Exception(f"Error in volatility prediction: {str(e)}")
    
    async def predict_portfolio_returns(self, symbols: List[str], weights: Optional[List[float]] = None, prediction_days: int = 30) -> Dict[str, Any]:
        """Predict portfolio returns"""
        try:
            if not weights:
                weights = [1/len(symbols)] * len(symbols)
            
            if len(symbols) != len(weights):
                raise ValueError("Number of symbols and weights must match")
            
            # Get data for all symbols
            portfolio_data = {}
            for symbol in symbols:
                data = await self._get_historical_data(symbol, "1y")
                if not data.empty:
                    portfolio_data[symbol] = data['Close'].pct_change().dropna()
            
            if len(portfolio_data) < 2:
                raise ValueError("At least 2 symbols required for portfolio prediction")
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, weights)
            
            # Predict future portfolio returns
            predicted_returns = await self._predict_portfolio_performance(portfolio_returns, prediction_days)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(portfolio_returns, predicted_returns)
            
            return {
                "symbols": symbols,
                "weights": weights,
                "prediction_days": prediction_days,
                "predicted_returns": predicted_returns,
                "portfolio_metrics": portfolio_metrics,
                "risk_metrics": self._calculate_portfolio_risk_metrics(portfolio_returns, predicted_returns),
                "optimization_suggestions": self._suggest_portfolio_optimization(portfolio_data, weights)
            }
        except Exception as e:
            raise Exception(f"Error in portfolio prediction: {str(e)}")
    
    async def predict_market_crash_probability(self) -> Dict[str, Any]:
        """Predict probability of market crash using multiple indicators"""
        try:
            # Get market data
            market_data = await self._get_market_indicators()
            
            # Calculate crash indicators
            indicators = self._calculate_crash_indicators(market_data)
            
            # Predict crash probability
            crash_probability = self._calculate_crash_probability(indicators)
            
            return {
                "crash_probability": crash_probability,
                "risk_level": self._assess_crash_risk(crash_probability),
                "indicators": indicators,
                "market_conditions": self._assess_market_conditions(market_data),
                "recommendations": self._generate_crash_recommendations(crash_probability, indicators)
            }
        except Exception as e:
            raise Exception(f"Error in crash prediction: {str(e)}")
    
    async def _get_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=timeframe)
            return data
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models"""
        features = data.copy()
        
        # Add technical indicators
        features = self._add_technical_indicators(features)
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['Volume'].shift(lag)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            features[f'close_ma_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'volume_ma_{window}'] = features['Volume'].rolling(window=window).mean()
            features[f'volatility_{window}'] = features['Close'].rolling(window=window).std()
        
        # Add price ratios
        features['price_to_ma20'] = features['Close'] / features['close_ma_20']
        features['volume_ratio'] = features['Volume'] / features['volume_ma_20']
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        try:
            import ta
            
            # Moving averages
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            
            # MACD
            data['MACD'] = ta.trend.macd_diff(data['Close'])
            data['MACD_signal'] = ta.trend.macd_signal(data['Close'])
            
            # RSI
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bb.bollinger_hband()
            data['BB_lower'] = bb.bollinger_lband()
            data['BB_middle'] = bb.bollinger_mavg()
            
            return data
        except Exception:
            return data
    
    async def _lstm_prediction(self, features: pd.DataFrame, prediction_days: int) -> np.ndarray:
        """LSTM-based price prediction (simplified implementation)"""
        try:
            # For a full LSTM implementation, you would use TensorFlow/PyTorch
            # Here we'll use a simplified approach with linear regression on technical indicators
            
            # Select relevant features
            feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'close_ma_5', 'close_ma_10']
            available_cols = [col for col in feature_cols if col in features.columns]
            
            if not available_cols:
                # Fallback to simple moving average
                return np.full(prediction_days, features['Close'].iloc[-1])
            
            X = features[available_cols].values
            y = features['Close'].values
            
            # Train linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            predictions = []
            last_features = X[-1].reshape(1, -1)
            
            for _ in range(prediction_days):
                pred = model.predict(last_features)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified)
                last_features = last_features * 0.99  # Simple decay
            
            return np.array(predictions)
        except Exception:
            # Fallback to simple trend continuation
            return np.full(prediction_days, features['Close'].iloc[-1])
    
    async def _arima_prediction(self, data: pd.DataFrame, prediction_days: int) -> np.ndarray:
        """ARIMA-based price prediction (simplified)"""
        try:
            # Simplified ARIMA implementation
            prices = data['Close'].values
            
            # Calculate trend
            x = np.arange(len(prices))
            trend = np.polyfit(x, prices, 1)[0]
            
            # Calculate seasonal component (simplified)
            seasonal = np.sin(2 * np.pi * np.arange(len(prices)) / 252) * prices.std() * 0.1
            
            # Make predictions
            predictions = []
            last_price = prices[-1]
            
            for i in range(prediction_days):
                # Simple trend + seasonal + noise
                pred = last_price + trend + seasonal[-1] * 0.1 + np.random.normal(0, prices.std() * 0.02)
                predictions.append(pred)
                last_price = pred
            
            return np.array(predictions)
        except Exception:
            return np.full(prediction_days, data['Close'].iloc[-1])
    
    async def _linear_regression_prediction(self, features: pd.DataFrame, prediction_days: int) -> np.ndarray:
        """Linear regression-based prediction"""
        try:
            # Select features
            feature_cols = [col for col in features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            if not feature_cols:
                return np.full(prediction_days, features['Close'].iloc[-1])
            
            X = features[feature_cols].values
            y = features['Close'].values
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions using last known features
            last_features = X[-1].reshape(1, -1)
            predictions = []
            
            for _ in range(prediction_days):
                pred = model.predict(last_features)[0]
                predictions.append(pred)
                # Simple feature update
                last_features = last_features * 0.995
            
            return np.array(predictions)
        except Exception:
            return np.full(prediction_days, features['Close'].iloc[-1])
    
    async def _ensemble_prediction(self, features: pd.DataFrame, data: pd.DataFrame, prediction_days: int) -> np.ndarray:
        """Ensemble prediction combining multiple models"""
        try:
            # Get predictions from different models
            lstm_pred = await self._lstm_prediction(features, prediction_days)
            arima_pred = await self._arima_prediction(data, prediction_days)
            lr_pred = await self._linear_regression_prediction(features, prediction_days)
            
            # Weighted average
            weights = [0.4, 0.3, 0.3]
            ensemble_pred = (lstm_pred * weights[0] + arima_pred * weights[1] + lr_pred * weights[2])
            
            return ensemble_pred
        except Exception:
            return np.full(prediction_days, data['Close'].iloc[-1])
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        try:
            # Calculate historical volatility
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate confidence intervals
            confidence_level = 0.95
            z_score = 1.96  # 95% confidence
            
            lower_bound = predictions * (1 - z_score * volatility)
            upper_bound = predictions * (1 + z_score * volatility)
            
            return {
                "lower": lower_bound,
                "upper": upper_bound
            }
        except Exception:
            return {
                "lower": predictions * 0.95,
                "upper": predictions * 1.05
            }
    
    def _calculate_model_accuracy(self, features: pd.DataFrame, data: pd.DataFrame) -> float:
        """Calculate model accuracy on historical data"""
        try:
            # Simple accuracy calculation based on trend direction
            actual_changes = data['Close'].pct_change().dropna()
            correct_direction = 0
            total_predictions = 0
            
            for i in range(10, len(actual_changes)):
                # Simple trend prediction
                recent_trend = actual_changes.iloc[i-10:i].mean()
                predicted_direction = 1 if recent_trend > 0 else -1
                actual_direction = 1 if actual_changes.iloc[i] > 0 else -1
                
                if predicted_direction == actual_direction:
                    correct_direction += 1
                total_predictions += 1
            
            return (correct_direction / total_predictions * 100) if total_predictions > 0 else 50
        except Exception:
            return 50
    
    def _determine_trend_direction(self, predictions: np.ndarray) -> str:
        """Determine overall trend direction from predictions"""
        if len(predictions) < 2:
            return "neutral"
        
        start_price = predictions[0]
        end_price = predictions[-1]
        change_percent = (end_price - start_price) / start_price * 100
        
        if change_percent > 5:
            return "strong_uptrend"
        elif change_percent > 1:
            return "uptrend"
        elif change_percent < -5:
            return "strong_downtrend"
        elif change_percent < -1:
            return "downtrend"
        else:
            return "sideways"
    
    def _assess_prediction_risk(self, predictions: np.ndarray, data: pd.DataFrame) -> str:
        """Assess risk level of predictions"""
        try:
            # Calculate prediction volatility
            pred_volatility = np.std(predictions) / np.mean(predictions)
            
            # Calculate historical volatility
            returns = data['Close'].pct_change().dropna()
            hist_volatility = returns.std()
            
            if pred_volatility > hist_volatility * 1.5:
                return "high"
            elif pred_volatility > hist_volatility * 1.2:
                return "medium"
            else:
                return "low"
        except Exception:
            return "medium"
    
    def _prepare_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for trend prediction"""
        features = data.copy()
        
        # Add trend indicators
        features['price_change'] = features['Close'].pct_change()
        features['volume_change'] = features['Volume'].pct_change()
        features['high_low_ratio'] = features['High'] / features['Low']
        features['close_open_ratio'] = features['Close'] / features['Open']
        
        # Add moving averages
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = features['Close'].rolling(window=window).mean()
            features[f'ma_ratio_{window}'] = features['Close'] / features[f'ma_{window}']
        
        return features.dropna()
    
    async def _predict_trend_direction(self, features: pd.DataFrame, prediction_days: int) -> Dict[str, Any]:
        """Predict trend direction using ensemble method"""
        try:
            # Select features for trend prediction
            feature_cols = [col for col in features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            if not feature_cols:
                return {"direction": "neutral", "confidence": 0.5}
            
            X = features[feature_cols].values
            y = (features['Close'].pct_change().shift(-1) > 0).astype(int).values[:-1]  # Next day direction
            
            if len(X) != len(y):
                X = X[:-1]
            
            # Train Random Forest for trend prediction
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Predict trend
            last_features = X[-1].reshape(1, -1)
            trend_probability = model.predict(last_features)[0]
            
            # Determine direction and confidence
            if trend_probability > 0.6:
                direction = "bullish"
                confidence = trend_probability
            elif trend_probability < 0.4:
                direction = "bearish"
                confidence = 1 - trend_probability
            else:
                direction = "neutral"
                confidence = 0.5
            
            return {
                "direction": direction,
                "confidence": confidence,
                "probability": trend_probability
            }
        except Exception:
            return {"direction": "neutral", "confidence": 0.5}
    
    def _calculate_trend_strength(self, trend_predictions: Dict, data: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            # Calculate recent volatility
            returns = data['Close'].pct_change().dropna()
            recent_volatility = returns.tail(20).std()
            
            # Calculate trend strength based on confidence and volatility
            base_strength = trend_predictions['confidence'] * 100
            volatility_adjustment = min(20, recent_volatility * 1000)
            
            strength = base_strength - volatility_adjustment
            return max(0, min(100, strength))
        except Exception:
            return 50
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identify key support and resistance levels"""
        try:
            prices = data['Close'].values
            
            # Find local maxima and minima
            from scipy.signal import argrelextrema
            
            maxima = argrelextrema(prices, np.greater, order=5)[0]
            minima = argrelextrema(prices, np.less, order=5)[0]
            
            # Get recent levels
            recent_maxima = prices[maxima[-3:]] if len(maxima) >= 3 else prices[maxima]
            recent_minima = prices[minima[-3:]] if len(minima) >= 3 else prices[minima]
            
            return {
                "resistance_levels": recent_maxima.tolist(),
                "support_levels": recent_minima.tolist(),
                "current_price": float(prices[-1])
            }
        except Exception:
            return {
                "resistance_levels": [],
                "support_levels": [],
                "current_price": float(data['Close'].iloc[-1])
            }
    
    def _find_support_resistance_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Find support and resistance levels"""
        try:
            # Simple support/resistance calculation
            high_20 = data['High'].rolling(window=20).max()
            low_20 = data['Low'].rolling(window=20).min()
            
            return {
                "resistance": float(high_20.iloc[-1]),
                "support": float(low_20.iloc[-1]),
                "current_price": float(data['Close'].iloc[-1])
            }
        except Exception:
            return {
                "resistance": float(data['Close'].iloc[-1] * 1.05),
                "support": float(data['Close'].iloc[-1] * 0.95),
                "current_price": float(data['Close'].iloc[-1])
            }
    
    def _forecast_volatility(self, data: pd.DataFrame, prediction_days: int) -> Dict[str, Any]:
        """Forecast volatility"""
        try:
            returns = data['Close'].pct_change().dropna()
            current_vol = returns.std() * np.sqrt(252)
            
            # Simple volatility forecast
            vol_forecast = current_vol * (1 + np.random.normal(0, 0.1, prediction_days))
            
            return {
                "current_volatility": current_vol,
                "forecasted_volatility": vol_forecast.tolist(),
                "trend": "increasing" if vol_forecast[-1] > vol_forecast[0] else "decreasing"
            }
        except Exception:
            return {
                "current_volatility": 0.2,
                "forecasted_volatility": [0.2] * prediction_days,
                "trend": "stable"
            }
    
    def _forecast_volatility_garch(self, returns: pd.Series) -> Dict[str, Any]:
        """Forecast volatility using GARCH-like approach"""
        try:
            # Simplified GARCH forecast
            current_vol = returns.std()
            vol_trend = returns.tail(20).std() - returns.tail(40).std()
            
            predicted_vol = current_vol * (1 + vol_trend * 0.1)
            
            return {
                "predicted": predicted_vol,
                "trend": "increasing" if vol_trend > 0 else "decreasing"
            }
        except Exception:
            return {
                "predicted": returns.std(),
                "trend": "stable"
            }
    
    def _calculate_volatility_percentiles(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility percentiles"""
        try:
            vol_252 = returns.std() * np.sqrt(252)
            
            return {
                "current": vol_252,
                "percentile_25": np.percentile(returns.rolling(252).std().dropna() * np.sqrt(252), 25),
                "percentile_50": np.percentile(returns.rolling(252).std().dropna() * np.sqrt(252), 50),
                "percentile_75": np.percentile(returns.rolling(252).std().dropna() * np.sqrt(252), 75),
                "percentile_90": np.percentile(returns.rolling(252).std().dropna() * np.sqrt(252), 90)
            }
        except Exception:
            return {
                "current": 0.2,
                "percentile_25": 0.15,
                "percentile_50": 0.2,
                "percentile_75": 0.25,
                "percentile_90": 0.3
            }
    
    def _assess_volatility_risk(self, predicted_volatility: float) -> str:
        """Assess volatility risk level"""
        if predicted_volatility > 0.4:
            return "very_high"
        elif predicted_volatility > 0.3:
            return "high"
        elif predicted_volatility > 0.2:
            return "medium"
        elif predicted_volatility > 0.1:
            return "low"
        else:
            return "very_low"
    
    def _calculate_portfolio_returns(self, portfolio_data: Dict[str, pd.Series], weights: List[float]) -> pd.Series:
        """Calculate portfolio returns"""
        try:
            # Align all return series
            aligned_data = pd.DataFrame(portfolio_data).fillna(0)
            
            # Calculate weighted portfolio returns
            portfolio_returns = (aligned_data * weights).sum(axis=1)
            
            return portfolio_returns
        except Exception:
            return pd.Series([0] * 100)
    
    async def _predict_portfolio_performance(self, portfolio_returns: pd.Series, prediction_days: int) -> Dict[str, Any]:
        """Predict portfolio performance"""
        try:
            # Simple prediction based on historical performance
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            
            # Generate predicted returns
            predicted_returns = np.random.normal(mean_return, std_return, prediction_days)
            
            return {
                "predicted_daily_returns": predicted_returns.tolist(),
                "expected_total_return": np.sum(predicted_returns),
                "expected_annual_return": mean_return * 252,
                "expected_volatility": std_return * np.sqrt(252)
            }
        except Exception:
            return {
                "predicted_daily_returns": [0] * prediction_days,
                "expected_total_return": 0,
                "expected_annual_return": 0,
                "expected_volatility": 0.2
            }
    
    def _calculate_portfolio_metrics(self, historical_returns: pd.Series, predicted_returns: Dict) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            # Historical metrics
            hist_mean = historical_returns.mean()
            hist_std = historical_returns.std()
            hist_sharpe = hist_mean / hist_std if hist_std > 0 else 0
            
            # Predicted metrics
            pred_mean = predicted_returns['expected_annual_return']
            pred_std = predicted_returns['expected_volatility']
            pred_sharpe = pred_mean / pred_std if pred_std > 0 else 0
            
            return {
                "historical_sharpe_ratio": hist_sharpe,
                "predicted_sharpe_ratio": pred_sharpe,
                "historical_volatility": hist_std * np.sqrt(252),
                "predicted_volatility": pred_std,
                "return_improvement": pred_sharpe - hist_sharpe
            }
        except Exception:
            return {
                "historical_sharpe_ratio": 0,
                "predicted_sharpe_ratio": 0,
                "historical_volatility": 0.2,
                "predicted_volatility": 0.2,
                "return_improvement": 0
            }
    
    def _calculate_portfolio_risk_metrics(self, historical_returns: pd.Series, predicted_returns: Dict) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        try:
            # Value at Risk
            var_95 = np.percentile(historical_returns, 5)
            var_99 = np.percentile(historical_returns, 1)
            
            # Maximum Drawdown
            cumulative = (1 + historical_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "max_drawdown": max_drawdown,
                "risk_score": abs(max_drawdown) * 100
            }
        except Exception:
            return {
                "var_95": -0.05,
                "var_99": -0.1,
                "max_drawdown": -0.2,
                "risk_score": 20
            }
    
    def _suggest_portfolio_optimization(self, portfolio_data: Dict[str, pd.Series], weights: List[float]) -> List[str]:
        """Suggest portfolio optimization strategies"""
        suggestions = []
        
        try:
            # Calculate individual asset metrics
            asset_metrics = {}
            for symbol, returns in portfolio_data.items():
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
                asset_metrics[symbol] = sharpe
            
            # Find best and worst performers
            best_asset = max(asset_metrics, key=asset_metrics.get)
            worst_asset = min(asset_metrics, key=asset_metrics.get)
            
            if asset_metrics[best_asset] > asset_metrics[worst_asset] * 1.5:
                suggestions.append(f"Consider increasing allocation to {best_asset}")
                suggestions.append(f"Consider reducing allocation to {worst_asset}")
            
            # Diversification suggestions
            if len(portfolio_data) < 5:
                suggestions.append("Consider adding more assets for better diversification")
            
            # Risk management suggestions
            total_volatility = sum(returns.std() * weight for returns, weight in zip(portfolio_data.values(), weights))
            if total_volatility > 0.3:
                suggestions.append("Portfolio volatility is high, consider risk management strategies")
            
        except Exception:
            suggestions.append("Unable to generate optimization suggestions")
        
        return suggestions
    
    async def _get_market_indicators(self) -> Dict[str, Any]:
        """Get market-wide indicators"""
        try:
            # Get major indices
            indices = ['^GSPC', '^VIX', '^TNX', '^DXY']
            market_data = {}
            
            for symbol in indices:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                if not data.empty:
                    market_data[symbol] = data['Close'].iloc[-1]
            
            return market_data
        except Exception:
            return {}
    
    def _calculate_crash_indicators(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate crash indicators"""
        try:
            indicators = {}
            
            # VIX (Volatility Index)
            vix = market_data.get('^VIX', 20)
            indicators['vix'] = vix
            indicators['vix_signal'] = 1 if vix > 30 else 0
            
            # Interest rates
            tnx = market_data.get('^TNX', 3)
            indicators['interest_rate'] = tnx
            indicators['rate_signal'] = 1 if tnx > 4 else 0
            
            # Dollar strength
            dxy = market_data.get('^DXY', 100)
            indicators['dollar_strength'] = dxy
            indicators['dollar_signal'] = 1 if dxy > 105 else 0
            
            return indicators
        except Exception:
            return {}
    
    def _calculate_crash_probability(self, indicators: Dict[str, float]) -> float:
        """Calculate crash probability based on indicators"""
        try:
            # Simple scoring system
            score = 0
            
            # VIX signal
            score += indicators.get('vix_signal', 0) * 30
            
            # Interest rate signal
            score += indicators.get('rate_signal', 0) * 25
            
            # Dollar strength signal
            score += indicators.get('dollar_signal', 0) * 20
            
            # Market momentum (simplified)
            score += np.random.uniform(0, 25)  # Placeholder for momentum calculation
            
            return min(100, max(0, score))
        except Exception:
            return 25
    
    def _assess_crash_risk(self, probability: float) -> str:
        """Assess crash risk level"""
        if probability > 70:
            return "very_high"
        elif probability > 50:
            return "high"
        elif probability > 30:
            return "medium"
        elif probability > 15:
            return "low"
        else:
            return "very_low"
    
    def _assess_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, str]:
        """Assess current market conditions"""
        try:
            conditions = {}
            
            # VIX assessment
            vix = market_data.get('^VIX', 20)
            if vix > 30:
                conditions['volatility'] = "high"
            elif vix > 20:
                conditions['volatility'] = "medium"
            else:
                conditions['volatility'] = "low"
            
            # Interest rate assessment
            tnx = market_data.get('^TNX', 3)
            if tnx > 4:
                conditions['interest_rates'] = "high"
            elif tnx > 2:
                conditions['interest_rates'] = "medium"
            else:
                conditions['interest_rates'] = "low"
            
            return conditions
        except Exception:
            return {"volatility": "medium", "interest_rates": "medium"}
    
    def _generate_crash_recommendations(self, probability: float, indicators: Dict[str, float]) -> List[str]:
        """Generate recommendations based on crash probability"""
        recommendations = []
        
        if probability > 60:
            recommendations.append("Consider reducing equity exposure")
            recommendations.append("Increase allocation to defensive assets")
            recommendations.append("Implement stop-loss orders")
        elif probability > 40:
            recommendations.append("Monitor market conditions closely")
            recommendations.append("Consider hedging strategies")
        else:
            recommendations.append("Market conditions appear stable")
            recommendations.append("Continue with current strategy")
        
        return recommendations
