import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
from textblob import TextBlob
import re
from datetime import datetime, timedelta
import yfinance as yf

class SentimentService:
    def __init__(self):
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://feeds.bloomberg.com/markets/news.rss"
        ]
    
    async def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Perform comprehensive sentiment analysis"""
        try:
            # Get news sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Get social media sentiment (simulated)
            social_sentiment = await self._analyze_social_sentiment(symbol)
            
            # Get analyst sentiment
            analyst_sentiment = await self._analyze_analyst_sentiment(symbol)
            
            # Calculate overall sentiment score
            overall_score = self._calculate_overall_sentiment_score(
                news_sentiment, social_sentiment, analyst_sentiment
            )
            
            return {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "news_sentiment": news_sentiment,
                "social_sentiment": social_sentiment,
                "analyst_sentiment": analyst_sentiment,
                "overall_sentiment_score": overall_score,
                "sentiment_trend": self._determine_sentiment_trend(overall_score)
            }
        except Exception as e:
            raise Exception(f"Error in sentiment analysis: {str(e)}")
    
    async def _analyze_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from financial news"""
        try:
            # Get recent news articles
            articles = await self._fetch_news_articles(symbol)
            
            if not articles:
                return {
                    "score": 50,
                    "sentiment": "neutral",
                    "article_count": 0,
                    "articles": []
                }
            
            # Analyze sentiment for each article
            sentiments = []
            analyzed_articles = []
            
            for article in articles:
                sentiment_score = self._calculate_text_sentiment(article.get('title', '') + ' ' + article.get('summary', ''))
                sentiments.append(sentiment_score)
                
                analyzed_articles.append({
                    "title": article.get('title', ''),
                    "summary": article.get('summary', ''),
                    "sentiment_score": sentiment_score,
                    "sentiment": self._score_to_sentiment(sentiment_score),
                    "published": article.get('published', ''),
                    "source": article.get('source', '')
                })
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 50
            
            return {
                "score": avg_sentiment,
                "sentiment": self._score_to_sentiment(avg_sentiment),
                "article_count": len(articles),
                "articles": analyzed_articles[:10]  # Return top 10 articles
            }
        except Exception as e:
            return {
                "score": 50,
                "sentiment": "neutral",
                "article_count": 0,
                "articles": [],
                "error": str(e)
            }
    
    async def _analyze_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from social media (simulated)"""
        try:
            # In a real implementation, this would connect to Twitter API, Reddit API, etc.
            # For now, we'll simulate based on stock performance and news sentiment
            
            # Get recent stock performance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d")
            
            if data.empty:
                return {
                    "score": 50,
                    "sentiment": "neutral",
                    "mentions": 0,
                    "trend": "stable"
                }
            
            # Calculate performance-based sentiment
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = (current_price - prev_price) / prev_price
            
            # Simulate social sentiment based on price movement
            base_score = 50
            sentiment_score = base_score + (price_change * 1000)  # Amplify price change
            sentiment_score = max(0, min(100, sentiment_score))
            
            # Simulate mention count based on volatility
            volatility = data['Close'].pct_change().std()
            mentions = int(volatility * 10000)  # Simulate mention count
            
            return {
                "score": sentiment_score,
                "sentiment": self._score_to_sentiment(sentiment_score),
                "mentions": mentions,
                "trend": "increasing" if price_change > 0.02 else "decreasing" if price_change < -0.02 else "stable"
            }
        except Exception as e:
            return {
                "score": 50,
                "sentiment": "neutral",
                "mentions": 0,
                "trend": "stable",
                "error": str(e)
            }
    
    async def _analyze_analyst_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze analyst recommendations and sentiment"""
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations.empty:
                return {
                    "score": 50,
                    "sentiment": "neutral",
                    "recommendations": [],
                    "consensus": "Hold"
                }
            
            # Get recent recommendations (last 3 months)
            recent_recommendations = recommendations.tail(20)
            
            # Analyze recommendation sentiment
            recommendation_scores = []
            analyzed_recommendations = []
            
            for _, rec in recent_recommendations.iterrows():
                firm = rec.get('firm', 'Unknown')
                to_grade = rec.get('toGrade', 'Hold')
                score = self._recommendation_to_score(to_grade)
                recommendation_scores.append(score)
                
                analyzed_recommendations.append({
                    "firm": firm,
                    "recommendation": to_grade,
                    "score": score,
                    "date": rec.name.isoformat() if hasattr(rec.name, 'isoformat') else str(rec.name)
                })
            
            # Calculate average sentiment
            avg_score = sum(recommendation_scores) / len(recommendation_scores) if recommendation_scores else 50
            
            # Determine consensus
            consensus = self._determine_consensus(recommendation_scores)
            
            return {
                "score": avg_score,
                "sentiment": self._score_to_sentiment(avg_score),
                "recommendations": analyzed_recommendations,
                "consensus": consensus,
                "recommendation_count": len(recommendation_scores)
            }
        except Exception as e:
            return {
                "score": 50,
                "sentiment": "neutral",
                "recommendations": [],
                "consensus": "Hold",
                "error": str(e)
            }
    
    async def _fetch_news_articles(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news articles for a given symbol"""
        try:
            # Use Yahoo Finance news
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            articles = []
            for article in news[:20]:  # Limit to 20 most recent articles
                articles.append({
                    "title": article.get('title', ''),
                    "summary": article.get('summary', ''),
                    "published": article.get('providerPublishTime', ''),
                    "source": article.get('publisher', ''),
                    "url": article.get('link', '')
                })
            
            return articles
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return []
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text using TextBlob"""
        try:
            if not text or not text.strip():
                return 50
            
            # Clean text
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Calculate sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert from [-1, 1] to [0, 100]
            score = (polarity + 1) * 50
            
            return max(0, min(100, score))
        except Exception:
            return 50
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label"""
        if score >= 70:
            return "very_positive"
        elif score >= 60:
            return "positive"
        elif score >= 40:
            return "neutral"
        elif score >= 30:
            return "negative"
        else:
            return "very_negative"
    
    def _recommendation_to_score(self, recommendation: str) -> float:
        """Convert analyst recommendation to numerical score"""
        recommendation = recommendation.lower()
        
        if recommendation in ['strong buy', 'buy']:
            return 80
        elif recommendation in ['outperform', 'positive']:
            return 70
        elif recommendation in ['hold', 'neutral', 'market perform']:
            return 50
        elif recommendation in ['underperform', 'negative']:
            return 30
        elif recommendation in ['sell', 'strong sell']:
            return 20
        else:
            return 50
    
    def _determine_consensus(self, scores: List[float]) -> str:
        """Determine consensus recommendation from scores"""
        if not scores:
            return "Hold"
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 70:
            return "Buy"
        elif avg_score >= 60:
            return "Outperform"
        elif avg_score >= 40:
            return "Hold"
        elif avg_score >= 30:
            return "Underperform"
        else:
            return "Sell"
    
    def _calculate_overall_sentiment_score(self, news: Dict, social: Dict, analyst: Dict) -> float:
        """Calculate weighted overall sentiment score"""
        weights = {
            "news": 0.4,
            "social": 0.3,
            "analyst": 0.3
        }
        
        overall_score = (
            news.get("score", 50) * weights["news"] +
            social.get("score", 50) * weights["social"] +
            analyst.get("score", 50) * weights["analyst"]
        )
        
        return round(overall_score, 2)
    
    def _determine_sentiment_trend(self, score: float) -> str:
        """Determine sentiment trend based on score"""
        if score >= 70:
            return "bullish"
        elif score >= 60:
            return "slightly_bullish"
        elif score >= 50:
            return "neutral"
        elif score >= 40:
            return "slightly_bearish"
        else:
            return "bearish"
