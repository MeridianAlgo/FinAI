import pandas as pd
import numpy as np
import asyncio
import aiohttp
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class DataTrainingService:
    def __init__(self):
        self.training_data = {}
        self.market_insights = {}
        self.company_database = {}
        self.news_sentiment_data = {}
        
    async def gather_comprehensive_training_data(self):
        """Gather comprehensive training data from multiple sources"""
        print("🚀 Starting comprehensive data gathering...")
        
        # Gather data from multiple sources
        await asyncio.gather(
            self._gather_stock_data(),
            self._gather_market_data(),
            self._gather_company_fundamentals(),
            self._gather_news_sentiment(),
            self._gather_technical_indicators(),
            self._gather_sector_analysis()
        )
        
        print("✅ Comprehensive training data gathered successfully")
        return self.training_data
    
    async def _gather_stock_data(self):
        """Gather historical stock data for training"""
        print("📊 Gathering stock data...")
        
        # Major stocks across different sectors
        stock_symbols = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC', 'CRM',
            'ADBE', 'PYPL', 'UBER', 'SPOT', 'ZM', 'SNOW', 'PLTR', 'CRWD', 'OKTA', 'ZS',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'MCO',
            'V', 'MA', 'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'NTRS', 'SCHW',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY',
            'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'MRNA', 'BNTX', 'ILMN', 'ISRG', 'DXCM',
            
            # Consumer
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'LOW', 'TGT', 'NKE', 'SBUX',
            'MCD', 'DIS', 'CMCSA', 'VZ', 'T', 'NFLX', 'AMZN', 'TSLA', 'F', 'GM',
            
            # Energy & Materials
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'KMI',
            'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'FCX', 'NEM', 'GOLD', 'AA',
            
            # Industrial
            'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'CMI', 'DE', 'CNHI', 'PCAR', 'DAL'
        ]
        
        stock_data = {}
        
        for symbol in stock_symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get historical data (2 years)
                hist_data = ticker.history(period="2y")
                
                if hist_data.empty:
                    continue
                
                # Get company info
                info = ticker.info
                
                # Calculate additional metrics
                hist_data = self._calculate_advanced_metrics(hist_data)
                
                stock_data[symbol] = {
                    'historical_data': hist_data,
                    'company_info': info,
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'last_updated': datetime.now().isoformat()
                }
                
                print(f"✅ Processed {symbol}")
                
            except Exception as e:
                print(f"⚠️ Error processing {symbol}: {str(e)}")
                continue
        
        self.training_data['stock_data'] = stock_data
        print(f"📊 Gathered data for {len(stock_data)} stocks")
    
    async def _gather_market_data(self):
        """Gather market-wide data and indices"""
        print("📈 Gathering market data...")
        
        market_indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Russell 2000': '^RUT',
            'VIX': '^VIX',
            'Dollar Index': '^DXY',
            '10-Year Treasury': '^TNX',
            '30-Year Treasury': '^TYX',
            'Gold': 'GC=F',
            'Oil': 'CL=F',
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD'
        }
        
        market_data = {}
        
        for name, symbol in market_indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period="1y")
                
                if not hist_data.empty:
                    market_data[name] = {
                        'symbol': symbol,
                        'data': hist_data,
                        'current_value': hist_data['Close'].iloc[-1],
                        'change_1d': hist_data['Close'].pct_change().iloc[-1],
                        'change_1w': hist_data['Close'].pct_change(5).iloc[-1],
                        'change_1m': hist_data['Close'].pct_change(20).iloc[-1],
                        'volatility': hist_data['Close'].pct_change().std() * np.sqrt(252)
                    }
                
            except Exception as e:
                print(f"⚠️ Error processing {name}: {str(e)}")
                continue
        
        self.training_data['market_data'] = market_data
        print(f"📈 Gathered data for {len(market_data)} market indicators")
    
    async def _gather_company_fundamentals(self):
        """Gather comprehensive fundamental data"""
        print("🏢 Gathering company fundamentals...")
        
        fundamental_data = {}
        
        # Get data for major companies
        major_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'WMT']
        
        for symbol in major_companies:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get financial statements
                financials = ticker.financials
                balance_sheet = ticker.balance_sheet
                cashflow = ticker.cashflow
                info = ticker.info
                
                # Calculate financial ratios
                ratios = self._calculate_financial_ratios(financials, balance_sheet, cashflow, info)
                
                fundamental_data[symbol] = {
                    'financials': financials.to_dict() if not financials.empty else {},
                    'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                    'cashflow': cashflow.to_dict() if not cashflow.empty else {},
                    'ratios': ratios,
                    'info': info
                }
                
            except Exception as e:
                print(f"⚠️ Error processing fundamentals for {symbol}: {str(e)}")
                continue
        
        self.training_data['fundamental_data'] = fundamental_data
        print(f"🏢 Gathered fundamentals for {len(fundamental_data)} companies")
    
    async def _gather_news_sentiment(self):
        """Gather news sentiment data"""
        print("📰 Gathering news sentiment data...")
        
        # This would typically involve web scraping news sites
        # For now, we'll simulate sentiment based on price movements
        sentiment_data = {}
        
        # Get news for major stocks
        major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        for symbol in major_stocks:
            try:
                ticker = yf.Ticker(symbol)
                news = ticker.news
                
                # Analyze sentiment from news headlines
                sentiment_scores = []
                for article in news[:20]:  # Last 20 articles
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    
                    # Simple sentiment analysis based on keywords
                    sentiment_score = self._analyze_text_sentiment(title + ' ' + summary)
                    sentiment_scores.append(sentiment_score)
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    sentiment_data[symbol] = {
                        'average_sentiment': avg_sentiment,
                        'sentiment_trend': 'positive' if avg_sentiment > 0.1 else 'negative' if avg_sentiment < -0.1 else 'neutral',
                        'news_count': len(sentiment_scores),
                        'last_updated': datetime.now().isoformat()
                    }
                
            except Exception as e:
                print(f"⚠️ Error processing news for {symbol}: {str(e)}")
                continue
        
        self.training_data['sentiment_data'] = sentiment_data
        print(f"📰 Gathered sentiment data for {len(sentiment_data)} stocks")
    
    async def _gather_technical_indicators(self):
        """Gather and calculate technical indicators"""
        print("📊 Gathering technical indicators...")
        
        technical_data = {}
        
        # Get data for major stocks
        major_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'WMT']
        
        for symbol in major_stocks:
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(period="1y")
                
                if not hist_data.empty:
                    # Calculate comprehensive technical indicators
                    technical_indicators = self._calculate_comprehensive_technical_indicators(hist_data)
                    
                    technical_data[symbol] = {
                        'indicators': technical_indicators,
                        'current_signals': self._generate_technical_signals(technical_indicators),
                        'last_updated': datetime.now().isoformat()
                    }
                
            except Exception as e:
                print(f"⚠️ Error processing technical indicators for {symbol}: {str(e)}")
                continue
        
        self.training_data['technical_data'] = technical_data
        print(f"📊 Gathered technical data for {len(technical_data)} stocks")
    
    async def _gather_sector_analysis(self):
        """Gather sector-wide analysis data"""
        print("🏭 Gathering sector analysis...")
        
        sectors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC', 'CRM'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'V', 'MA'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY'],
            'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'LOW', 'TGT', 'NKE', 'SBUX'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'KMI'],
            'Industrial': ['BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC']
        }
        
        sector_data = {}
        
        for sector_name, symbols in sectors.items():
            try:
                sector_performance = []
                sector_volatility = []
                sector_volume = []
                
                for symbol in symbols:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist_data = ticker.history(period="3mo")
                        
                        if not hist_data.empty:
                            # Calculate performance metrics
                            performance = hist_data['Close'].pct_change().sum()
                            volatility = hist_data['Close'].pct_change().std() * np.sqrt(252)
                            avg_volume = hist_data['Volume'].mean()
                            
                            sector_performance.append(performance)
                            sector_volatility.append(volatility)
                            sector_volume.append(avg_volume)
                    
                    except Exception:
                        continue
                
                if sector_performance:
                    sector_data[sector_name] = {
                        'average_performance': np.mean(sector_performance),
                        'average_volatility': np.mean(sector_volatility),
                        'average_volume': np.mean(sector_volume),
                        'performance_std': np.std(sector_performance),
                        'stock_count': len(sector_performance),
                        'top_performers': self._get_top_performers(symbols, sector_performance),
                        'last_updated': datetime.now().isoformat()
                    }
                
            except Exception as e:
                print(f"⚠️ Error processing sector {sector_name}: {str(e)}")
                continue
        
        self.training_data['sector_data'] = sector_data
        print(f"🏭 Gathered data for {len(sector_data)} sectors")
    
    def _calculate_advanced_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced financial metrics"""
        try:
            # Basic technical indicators
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
            
            # Support and Resistance
            data['Resistance'] = data['High'].rolling(window=20).max()
            data['Support'] = data['Low'].rolling(window=20).min()
            
            # Trend indicators
            data['Trend_strength'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
            data['Trend_direction'] = np.where(data['Close'] > data['SMA_20'], 1, -1)
            
            return data
            
        except Exception as e:
            print(f"⚠️ Error calculating advanced metrics: {str(e)}")
            return data
    
    def _calculate_financial_ratios(self, financials, balance_sheet, cashflow, info):
        """Calculate comprehensive financial ratios"""
        try:
            ratios = {}
            
            # Valuation ratios
            ratios['pe_ratio'] = info.get('trailingPE', 0)
            ratios['forward_pe'] = info.get('forwardPE', 0)
            ratios['peg_ratio'] = info.get('pegRatio', 0)
            ratios['price_to_book'] = info.get('priceToBook', 0)
            ratios['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
            ratios['price_to_cashflow'] = info.get('priceToCashflow', 0)
            
            # Profitability ratios
            ratios['gross_margin'] = info.get('grossMargins', 0)
            ratios['operating_margin'] = info.get('operatingMargins', 0)
            ratios['profit_margin'] = info.get('profitMargins', 0)
            ratios['return_on_equity'] = info.get('returnOnEquity', 0)
            ratios['return_on_assets'] = info.get('returnOnAssets', 0)
            ratios['return_on_invested_capital'] = info.get('returnOnInvestedCapital', 0)
            
            # Liquidity ratios
            ratios['current_ratio'] = info.get('currentRatio', 0)
            ratios['quick_ratio'] = info.get('quickRatio', 0)
            ratios['cash_ratio'] = info.get('totalCash', 0) / info.get('totalCurrentLiabilities', 1)
            
            # Leverage ratios
            ratios['debt_to_equity'] = info.get('debtToEquity', 0)
            ratios['debt_to_assets'] = info.get('totalDebt', 0) / info.get('totalAssets', 1)
            ratios['interest_coverage'] = info.get('operatingCashflow', 0) / max(info.get('interestExpense', 1), 1)
            
            # Efficiency ratios
            ratios['asset_turnover'] = info.get('revenue', 0) / info.get('totalAssets', 1)
            ratios['inventory_turnover'] = info.get('costOfRevenue', 0) / info.get('inventory', 1)
            ratios['receivables_turnover'] = info.get('revenue', 0) / info.get('netReceivables', 1)
            
            # Growth ratios
            ratios['revenue_growth'] = info.get('revenueGrowth', 0)
            ratios['earnings_growth'] = info.get('earningsGrowth', 0)
            ratios['book_value_growth'] = info.get('bookValue', 0) / info.get('bookValue', 1) - 1
            
            # Dividend ratios
            ratios['dividend_yield'] = info.get('dividendYield', 0)
            ratios['payout_ratio'] = info.get('payoutRatio', 0)
            ratios['dividend_growth'] = info.get('dividendRate', 0) / max(info.get('dividendRate', 1), 1) - 1
            
            return ratios
            
        except Exception as e:
            print(f"⚠️ Error calculating financial ratios: {str(e)}")
            return {}
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keywords"""
        try:
            if not text:
                return 0.0
            
            # Positive keywords
            positive_words = [
                'bullish', 'growth', 'profit', 'gain', 'rise', 'increase', 'strong', 'positive',
                'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'recommend', 'success',
                'breakthrough', 'innovation', 'leading', 'dominant', 'expansion'
            ]
            
            # Negative keywords
            negative_words = [
                'bearish', 'decline', 'loss', 'fall', 'decrease', 'weak', 'negative',
                'miss', 'underperform', 'downgrade', 'sell', 'avoid', 'failure',
                'crisis', 'recession', 'bankruptcy', 'lawsuit', 'investigation'
            ]
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment_score = (positive_count - negative_count) / total_words
            return max(-1, min(1, sentiment_score))
            
        except Exception:
            return 0.0
    
    def _calculate_comprehensive_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = data['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = data['Close'].rolling(window=50).mean().iloc[-1]
            indicators['sma_200'] = data['Close'].rolling(window=200).mean().iloc[-1]
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = macd_signal.iloc[-1]
            indicators['macd_histogram'] = (macd - macd_signal).iloc[-1]
            
            # Bollinger Bands
            bb_middle = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            indicators['bb_upper'] = bb_upper.iloc[-1]
            indicators['bb_middle'] = bb_middle.iloc[-1]
            indicators['bb_lower'] = bb_lower.iloc[-1]
            indicators['bb_position'] = (data['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            
            # Stochastic
            low_14 = data['Low'].rolling(window=14).min()
            high_14 = data['High'].rolling(window=14).max()
            stoch_k = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
            stoch_d = stoch_k.rolling(window=3).mean()
            indicators['stoch_k'] = stoch_k.iloc[-1]
            indicators['stoch_d'] = stoch_d.iloc[-1]
            
            # Williams %R
            indicators['williams_r'] = -100 * ((high_14.iloc[-1] - data['Close'].iloc[-1]) / (high_14.iloc[-1] - low_14.iloc[-1]))
            
            # ATR
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = data['Volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = data['Volume'].iloc[-1] / indicators['volume_sma']
            
            # Momentum
            indicators['momentum_1'] = data['Close'].pct_change(1).iloc[-1]
            indicators['momentum_5'] = data['Close'].pct_change(5).iloc[-1]
            indicators['momentum_10'] = data['Close'].pct_change(10).iloc[-1]
            indicators['momentum_20'] = data['Close'].pct_change(20).iloc[-1]
            
            # Volatility
            indicators['volatility_10'] = data['Close'].pct_change().rolling(window=10).std().iloc[-1]
            indicators['volatility_20'] = data['Close'].pct_change().rolling(window=20).std().iloc[-1]
            indicators['volatility_30'] = data['Close'].pct_change().rolling(window=30).std().iloc[-1]
            
            return indicators
            
        except Exception as e:
            print(f"⚠️ Error calculating technical indicators: {str(e)}")
            return {}
    
    def _generate_technical_signals(self, indicators: Dict) -> Dict:
        """Generate technical trading signals"""
        try:
            signals = {}
            
            # Moving average signals
            if indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
                signals['ma_signal'] = 'bullish'
            elif indicators.get('sma_20', 0) < indicators.get('sma_50', 0):
                signals['ma_signal'] = 'bearish'
            else:
                signals['ma_signal'] = 'neutral'
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                signals['rsi_signal'] = 'overbought'
            elif rsi < 30:
                signals['rsi_signal'] = 'oversold'
            else:
                signals['rsi_signal'] = 'neutral'
            
            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                signals['macd_signal'] = 'bullish'
            elif macd < macd_signal:
                signals['macd_signal'] = 'bearish'
            else:
                signals['macd_signal'] = 'neutral'
            
            # Bollinger Bands signals
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position > 0.8:
                signals['bb_signal'] = 'overbought'
            elif bb_position < 0.2:
                signals['bb_signal'] = 'oversold'
            else:
                signals['bb_signal'] = 'neutral'
            
            # Stochastic signals
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            if stoch_k > 80 and stoch_d > 80:
                signals['stoch_signal'] = 'overbought'
            elif stoch_k < 20 and stoch_d < 20:
                signals['stoch_signal'] = 'oversold'
            else:
                signals['stoch_signal'] = 'neutral'
            
            # Overall signal
            bullish_signals = sum(1 for signal in signals.values() if 'bullish' in signal or 'oversold' in signal)
            bearish_signals = sum(1 for signal in signals.values() if 'bearish' in signal or 'overbought' in signal)
            
            if bullish_signals > bearish_signals:
                signals['overall_signal'] = 'bullish'
            elif bearish_signals > bullish_signals:
                signals['overall_signal'] = 'bearish'
            else:
                signals['overall_signal'] = 'neutral'
            
            return signals
            
        except Exception as e:
            print(f"⚠️ Error generating technical signals: {str(e)}")
            return {}
    
    def _get_top_performers(self, symbols: List[str], performances: List[float]) -> List[Dict]:
        """Get top performing stocks in a sector"""
        try:
            # Combine symbols with performances
            stock_performances = list(zip(symbols, performances))
            
            # Sort by performance (descending)
            stock_performances.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 3 performers
            top_performers = []
            for symbol, performance in stock_performances[:3]:
                top_performers.append({
                    'symbol': symbol,
                    'performance': performance,
                    'performance_pct': performance * 100
                })
            
            return top_performers
            
        except Exception as e:
            print(f"⚠️ Error getting top performers: {str(e)}")
            return []
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of gathered training data"""
        try:
            summary = {
                'total_stocks': len(self.training_data.get('stock_data', {})),
                'market_indicators': len(self.training_data.get('market_data', {})),
                'fundamental_companies': len(self.training_data.get('fundamental_data', {})),
                'sentiment_stocks': len(self.training_data.get('sentiment_data', {})),
                'technical_stocks': len(self.training_data.get('technical_data', {})),
                'sectors_analyzed': len(self.training_data.get('sector_data', {})),
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Error generating training summary: {str(e)}")
            return {}
