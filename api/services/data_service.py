import yfinance as yf
import pandas as pd
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
import aiohttp

class DataService:
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    async def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            return data
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get additional data
            recommendations = ticker.recommendations
            calendar = ticker.calendar
            actions = ticker.actions
            
            return {
                "symbol": symbol,
                "info": info,
                "recommendations": recommendations.to_dict('records') if not recommendations.empty else [],
                "calendar": calendar.to_dict('records') if not calendar.empty else [],
                "actions": actions.to_dict('records') if not actions.empty else []
            }
        except Exception as e:
            raise Exception(f"Error fetching info for {symbol}: {str(e)}")
    
    async def get_trending_stocks(self) -> List[Dict[str, Any]]:
        """Get trending stocks from major indices"""
        try:
            # Major indices symbols
            indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
            trending = []
            
            for index in indices:
                ticker = yf.Ticker(index)
                data = ticker.history(period="5d")
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                    
                    trending.append({
                        "symbol": index,
                        "name": self._get_index_name(index),
                        "price": current_price,
                        "change_percent": change_percent,
                        "volume": data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                    })
            
            return sorted(trending, key=lambda x: abs(x['change_percent']), reverse=True)
        except Exception as e:
            raise Exception(f"Error fetching trending stocks: {str(e)}")
    
    async def get_market_indices(self) -> Dict[str, Any]:
        """Get major market indices data"""
        try:
            indices = {
                'S&P 500': '^GSPC',
                'Dow Jones': '^DJI',
                'NASDAQ': '^IXIC',
                'Russell 2000': '^RUT',
                'VIX': '^VIX'
            }
            
            market_data = {}
            
            for name, symbol in indices.items():
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2d")
                
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2] if len(data) > 1 else current
                    change = current - previous
                    change_percent = (change / previous) * 100
                    
                    market_data[name] = {
                        "symbol": symbol,
                        "price": current,
                        "change": change,
                        "change_percent": change_percent,
                        "volume": data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
                    }
            
            return market_data
        except Exception as e:
            raise Exception(f"Error fetching market indices: {str(e)}")
    
    async def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare multiple stocks"""
        try:
            comparison_data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                info = ticker.info
                
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    year_high = data['High'].max()
                    year_low = data['Low'].min()
                    avg_volume = data['Volume'].mean()
                    
                    # Calculate returns
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                    sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
                    
                    comparison_data[symbol] = {
                        "current_price": current_price,
                        "year_high": year_high,
                        "year_low": year_low,
                        "avg_volume": avg_volume,
                        "volatility": volatility,
                        "sharpe_ratio": sharpe_ratio,
                        "market_cap": info.get('marketCap', 0),
                        "pe_ratio": info.get('trailingPE', 0),
                        "sector": info.get('sector', 'Unknown')
                    }
            
            return comparison_data
        except Exception as e:
            raise Exception(f"Error comparing stocks: {str(e)}")
    
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
            
            # Volume indicators
            if 'Volume' in data.columns:
                data['Volume_SMA'] = ta.volume.volume_sma(data['Close'], data['Volume'])
            
            return data
        except Exception as e:
            print(f"Warning: Could not add technical indicators: {str(e)}")
            return data
    
    def _get_index_name(self, symbol: str) -> str:
        """Get human-readable name for index symbols"""
        names = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones Industrial Average',
            '^IXIC': 'NASDAQ Composite',
            '^RUT': 'Russell 2000'
        }
        return names.get(symbol, symbol)
