import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
from .data_service import DataService
from .sentiment_service import SentimentService

class AnalysisService:
    def __init__(self):
        self.data_service = DataService()
        self.sentiment_service = SentimentService()
    
    async def comprehensive_analysis(self, symbol: str, timeframe: str = "1y", analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform comprehensive financial analysis"""
        try:
            # Get stock data
            data = await self.data_service.get_stock_data(symbol, timeframe)
            
            # Perform different types of analysis
            technical_analysis = await self.technical_analysis(symbol, timeframe)
            fundamental_analysis = await self.fundamental_analysis(symbol)
            sentiment_analysis = await self.sentiment_service.analyze_sentiment(symbol)
            risk_analysis = await self.risk_analysis(symbol)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                technical_analysis, fundamental_analysis, sentiment_analysis, risk_analysis
            )
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis_date": datetime.now().isoformat(),
                "overall_score": overall_score,
                "technical_analysis": technical_analysis,
                "fundamental_analysis": fundamental_analysis,
                "sentiment_analysis": sentiment_analysis,
                "risk_analysis": risk_analysis,
                "recommendation": self._generate_recommendation(overall_score)
            }
        except Exception as e:
            raise Exception(f"Error in comprehensive analysis: {str(e)}")
    
    async def technical_analysis(self, symbol: str, timeframe: str = "1y", indicators: str = "all") -> Dict[str, Any]:
        """Perform technical analysis"""
        try:
            data = await self.data_service.get_stock_data(symbol, timeframe)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate technical indicators
            analysis = {
                "current_price": current_price,
                "price_change_1d": self._calculate_price_change(data, 1),
                "price_change_7d": self._calculate_price_change(data, 7),
                "price_change_30d": self._calculate_price_change(data, 30),
                "moving_averages": self._analyze_moving_averages(data),
                "macd_signal": self._analyze_macd(data),
                "rsi_analysis": self._analyze_rsi(data),
                "bollinger_bands": self._analyze_bollinger_bands(data),
                "support_resistance": self._find_support_resistance(data),
                "trend_analysis": self._analyze_trend(data),
                "volume_analysis": self._analyze_volume(data)
            }
            
            # Generate technical score
            analysis["technical_score"] = self._calculate_technical_score(analysis)
            
            return analysis
        except Exception as e:
            raise Exception(f"Error in technical analysis: {str(e)}")
    
    async def fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform fundamental analysis"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Key financial metrics
            analysis = {
                "valuation_metrics": {
                    "pe_ratio": info.get('trailingPE', 0),
                    "forward_pe": info.get('forwardPE', 0),
                    "peg_ratio": info.get('pegRatio', 0),
                    "price_to_book": info.get('priceToBook', 0),
                    "price_to_sales": info.get('priceToSalesTrailing12Months', 0),
                    "enterprise_value": info.get('enterpriseValue', 0),
                    "market_cap": info.get('marketCap', 0)
                },
                "profitability_metrics": {
                    "gross_margin": info.get('grossMargins', 0),
                    "operating_margin": info.get('operatingMargins', 0),
                    "profit_margin": info.get('profitMargins', 0),
                    "return_on_equity": info.get('returnOnEquity', 0),
                    "return_on_assets": info.get('returnOnAssets', 0)
                },
                "growth_metrics": {
                    "revenue_growth": info.get('revenueGrowth', 0),
                    "earnings_growth": info.get('earningsGrowth', 0),
                    "quarterly_revenue_growth": info.get('quarterlyRevenueGrowth', 0),
                    "quarterly_earnings_growth": info.get('quarterlyEarningsGrowth', 0)
                },
                "financial_health": {
                    "debt_to_equity": info.get('debtToEquity', 0),
                    "current_ratio": info.get('currentRatio', 0),
                    "quick_ratio": info.get('quickRatio', 0),
                    "cash_per_share": info.get('totalCashPerShare', 0)
                },
                "dividend_info": {
                    "dividend_yield": info.get('dividendYield', 0),
                    "dividend_rate": info.get('dividendRate', 0),
                    "payout_ratio": info.get('payoutRatio', 0)
                }
            }
            
            # Calculate fundamental score
            analysis["fundamental_score"] = self._calculate_fundamental_score(analysis)
            
            return analysis
        except Exception as e:
            raise Exception(f"Error in fundamental analysis: {str(e)}")
    
    async def risk_analysis(self, symbol: str) -> Dict[str, Any]:
        """Perform risk assessment analysis"""
        try:
            data = await self.data_service.get_stock_data(symbol, "1y")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            returns = data['Close'].pct_change().dropna()
            
            # Calculate risk metrics
            analysis = {
                "volatility_metrics": {
                    "daily_volatility": returns.std(),
                    "annualized_volatility": returns.std() * (252 ** 0.5),
                    "max_drawdown": self._calculate_max_drawdown(data['Close']),
                    "var_95": np.percentile(returns, 5),  # Value at Risk 95%
                    "var_99": np.percentile(returns, 1)   # Value at Risk 99%
                },
                "beta_analysis": await self._calculate_beta(symbol, data),
                "correlation_analysis": await self._calculate_correlations(symbol, data),
                "risk_score": 0  # Will be calculated
            }
            
            # Calculate overall risk score
            analysis["risk_score"] = self._calculate_risk_score(analysis)
            
            return analysis
        except Exception as e:
            raise Exception(f"Error in risk analysis: {str(e)}")
    
    async def portfolio_optimization(self, symbols: List[str], risk_tolerance: str = "medium") -> Dict[str, Any]:
        """Optimize portfolio allocation using Modern Portfolio Theory"""
        try:
            # Get data for all symbols
            portfolio_data = {}
            for symbol in symbols:
                data = await self.data_service.get_stock_data(symbol, "1y")
                if not data.empty:
                    portfolio_data[symbol] = data['Close'].pct_change().dropna()
            
            if len(portfolio_data) < 2:
                raise ValueError("At least 2 symbols required for portfolio optimization")
            
            # Calculate expected returns and covariance matrix
            returns_df = pd.DataFrame(portfolio_data)
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Portfolio optimization
            optimal_weights = self._optimize_portfolio(expected_returns, cov_matrix, risk_tolerance)
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                "symbols": symbols,
                "optimal_weights": dict(zip(symbols, optimal_weights)),
                "expected_return": portfolio_return,
                "expected_volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "risk_tolerance": risk_tolerance
            }
        except Exception as e:
            raise Exception(f"Error in portfolio optimization: {str(e)}")
    
    def _calculate_price_change(self, data: pd.DataFrame, days: int) -> Dict[str, float]:
        """Calculate price change over specified days"""
        if len(data) < days + 1:
            return {"change": 0, "change_percent": 0}
        
        current_price = data['Close'].iloc[-1]
        past_price = data['Close'].iloc[-(days + 1)]
        change = current_price - past_price
        change_percent = (change / past_price) * 100
        
        return {"change": change, "change_percent": change_percent}
    
    def _analyze_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze moving average signals"""
        if 'SMA_20' not in data.columns or 'SMA_50' not in data.columns:
            return {"signal": "neutral", "strength": 0}
        
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        # Determine signal
        if current_price > sma_20 > sma_50:
            signal = "bullish"
            strength = min(100, abs(current_price - sma_50) / sma_50 * 1000)
        elif current_price < sma_20 < sma_50:
            signal = "bearish"
            strength = min(100, abs(current_price - sma_50) / sma_50 * 1000)
        else:
            signal = "neutral"
            strength = 0
        
        return {
            "signal": signal,
            "strength": strength,
            "current_price": current_price,
            "sma_20": sma_20,
            "sma_50": sma_50
        }
    
    def _analyze_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze MACD signals"""
        if 'MACD' not in data.columns or 'MACD_signal' not in data.columns:
            return {"signal": "neutral", "strength": 0}
        
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        
        if macd > macd_signal:
            signal = "bullish"
            strength = min(100, abs(macd - macd_signal) * 1000)
        elif macd < macd_signal:
            signal = "bearish"
            strength = min(100, abs(macd - macd_signal) * 1000)
        else:
            signal = "neutral"
            strength = 0
        
        return {
            "signal": signal,
            "strength": strength,
            "macd": macd,
            "macd_signal": macd_signal
        }
    
    def _analyze_rsi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze RSI signals"""
        if 'RSI' not in data.columns:
            return {"signal": "neutral", "strength": 0}
        
        rsi = data['RSI'].iloc[-1]
        
        if rsi > 70:
            signal = "overbought"
            strength = min(100, (rsi - 70) * 3.33)
        elif rsi < 30:
            signal = "oversold"
            strength = min(100, (30 - rsi) * 3.33)
        else:
            signal = "neutral"
            strength = 0
        
        return {
            "signal": signal,
            "strength": strength,
            "rsi": rsi
        }
    
    def _analyze_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Bollinger Bands signals"""
        if 'BB_upper' not in data.columns or 'BB_lower' not in data.columns:
            return {"signal": "neutral", "strength": 0}
        
        current_price = data['Close'].iloc[-1]
        bb_upper = data['BB_upper'].iloc[-1]
        bb_lower = data['BB_lower'].iloc[-1]
        bb_middle = data['BB_middle'].iloc[-1]
        
        if current_price > bb_upper:
            signal = "overbought"
            strength = min(100, (current_price - bb_upper) / bb_upper * 1000)
        elif current_price < bb_lower:
            signal = "oversold"
            strength = min(100, (bb_lower - current_price) / bb_lower * 1000)
        else:
            signal = "neutral"
            strength = 0
        
        return {
            "signal": signal,
            "strength": strength,
            "current_price": current_price,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_middle
        }
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Find support and resistance levels"""
        prices = data['Close'].values
        
        # Simple support/resistance detection
        highs = data['High'].rolling(window=20).max()
        lows = data['Low'].rolling(window=20).min()
        
        resistance = highs.iloc[-1]
        support = lows.iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        return {
            "support": support,
            "resistance": resistance,
            "current_price": current_price,
            "distance_to_support": ((current_price - support) / support) * 100,
            "distance_to_resistance": ((resistance - current_price) / current_price) * 100
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall trend"""
        if len(data) < 50:
            return {"trend": "insufficient_data", "strength": 0}
        
        # Calculate trend using linear regression
        x = np.arange(len(data))
        y = data['Close'].values
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize slope to percentage
        trend_strength = (slope / y[0]) * 100 * len(data)
        
        if trend_strength > 5:
            trend = "strong_uptrend"
        elif trend_strength > 1:
            trend = "uptrend"
        elif trend_strength < -5:
            trend = "strong_downtrend"
        elif trend_strength < -1:
            trend = "downtrend"
        else:
            trend = "sideways"
        
        return {
            "trend": trend,
            "strength": abs(trend_strength),
            "slope": slope
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        if 'Volume' not in data.columns:
            return {"signal": "no_data", "strength": 0}
        
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 2:
            signal = "high_volume"
            strength = min(100, (volume_ratio - 1) * 50)
        elif volume_ratio < 0.5:
            signal = "low_volume"
            strength = min(100, (1 - volume_ratio) * 100)
        else:
            signal = "normal_volume"
            strength = 0
        
        return {
            "signal": signal,
            "strength": strength,
            "current_volume": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio
        }
    
    def _calculate_technical_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall technical analysis score"""
        score = 50  # Base score
        
        # Moving averages
        ma_signal = analysis["moving_averages"]["signal"]
        if ma_signal == "bullish":
            score += 15
        elif ma_signal == "bearish":
            score -= 15
        
        # MACD
        macd_signal = analysis["macd_signal"]["signal"]
        if macd_signal == "bullish":
            score += 10
        elif macd_signal == "bearish":
            score -= 10
        
        # RSI
        rsi_signal = analysis["rsi_analysis"]["signal"]
        if rsi_signal == "oversold":
            score += 10
        elif rsi_signal == "overbought":
            score -= 10
        
        # Trend
        trend = analysis["trend_analysis"]["trend"]
        if "uptrend" in trend:
            score += 10
        elif "downtrend" in trend:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_fundamental_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall fundamental analysis score"""
        score = 50  # Base score
        
        # Valuation metrics
        pe_ratio = analysis["valuation_metrics"]["pe_ratio"]
        if 10 < pe_ratio < 25:
            score += 10
        elif pe_ratio > 30:
            score -= 10
        
        # Profitability
        profit_margin = analysis["profitability_metrics"]["profit_margin"]
        if profit_margin > 0.1:
            score += 15
        elif profit_margin < 0:
            score -= 20
        
        # Growth
        revenue_growth = analysis["growth_metrics"]["revenue_growth"]
        if revenue_growth > 0.1:
            score += 10
        elif revenue_growth < -0.1:
            score -= 10
        
        # Financial health
        debt_to_equity = analysis["financial_health"]["debt_to_equity"]
        if debt_to_equity < 0.5:
            score += 10
        elif debt_to_equity > 1:
            score -= 10
        
        return max(0, min(100, score))
    
    def _calculate_risk_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate risk score (higher = more risky)"""
        score = 50  # Base score
        
        # Volatility
        volatility = analysis["volatility_metrics"]["annualized_volatility"]
        if volatility > 0.3:
            score += 20
        elif volatility < 0.15:
            score -= 10
        
        # Max drawdown
        max_drawdown = analysis["volatility_metrics"]["max_drawdown"]
        if max_drawdown > 0.3:
            score += 15
        elif max_drawdown < 0.1:
            score -= 5
        
        # Beta
        beta = analysis["beta_analysis"].get("beta", 1)
        if beta > 1.5:
            score += 15
        elif beta < 0.5:
            score -= 5
        
        return max(0, min(100, score))
    
    def _calculate_overall_score(self, technical: Dict, fundamental: Dict, sentiment: Dict, risk: Dict) -> float:
        """Calculate overall analysis score"""
        weights = {
            "technical": 0.3,
            "fundamental": 0.3,
            "sentiment": 0.2,
            "risk": 0.2
        }
        
        # Risk score is inverted (lower risk = higher score)
        risk_score = 100 - risk.get("risk_score", 50)
        sentiment_score = sentiment.get("overall_sentiment_score", 50)
        
        overall_score = (
            technical.get("technical_score", 50) * weights["technical"] +
            fundamental.get("fundamental_score", 50) * weights["fundamental"] +
            sentiment_score * weights["sentiment"] +
            risk_score * weights["risk"]
        )
        
        return round(overall_score, 2)
    
    def _generate_recommendation(self, overall_score: float) -> str:
        """Generate investment recommendation based on overall score"""
        if overall_score >= 80:
            return "Strong Buy"
        elif overall_score >= 65:
            return "Buy"
        elif overall_score >= 50:
            return "Hold"
        elif overall_score >= 35:
            return "Sell"
        else:
            return "Strong Sell"
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())
    
    async def _calculate_beta(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate beta relative to market"""
        try:
            # Get S&P 500 data for comparison
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="1y")
            
            if sp500_data.empty or data.empty:
                return {"beta": 1.0, "correlation": 0.0}
            
            # Align dates
            common_dates = data.index.intersection(sp500_data.index)
            if len(common_dates) < 30:
                return {"beta": 1.0, "correlation": 0.0}
            
            stock_returns = data.loc[common_dates, 'Close'].pct_change().dropna()
            market_returns = sp500_data.loc[common_dates, 'Close'].pct_change().dropna()
            
            # Calculate beta and correlation
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
            
            return {"beta": beta, "correlation": correlation}
        except Exception:
            return {"beta": 1.0, "correlation": 0.0}
    
    async def _calculate_correlations(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations with major indices"""
        try:
            indices = {
                "S&P 500": "^GSPC",
                "NASDAQ": "^IXIC",
                "Dow Jones": "^DJI"
            }
            
            correlations = {}
            stock_returns = data['Close'].pct_change().dropna()
            
            for name, ticker_symbol in indices.items():
                try:
                    index_data = yf.Ticker(ticker_symbol).history(period="1y")
                    if not index_data.empty:
                        common_dates = data.index.intersection(index_data.index)
                        if len(common_dates) > 30:
                            index_returns = index_data.loc[common_dates, 'Close'].pct_change().dropna()
                            stock_returns_aligned = stock_returns.loc[common_dates]
                            
                            correlation = np.corrcoef(stock_returns_aligned, index_returns)[0, 1]
                            correlations[name] = correlation
                except Exception:
                    correlations[name] = 0.0
            
            return correlations
        except Exception:
            return {}
    
    def _optimize_portfolio(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_tolerance: str) -> np.ndarray:
        """Optimize portfolio using Modern Portfolio Theory"""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            # Risk tolerance parameters
            risk_params = {
                "low": 0.1,
                "medium": 0.15,
                "high": 0.25
            }
            target_volatility = risk_params.get(risk_tolerance, 0.15)
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds: weights between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            return result.x if result.success else x0
        except Exception:
            # Fallback to equal weights
            return np.array([1/len(expected_returns)] * len(expected_returns))
