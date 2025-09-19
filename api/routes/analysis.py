from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from ..services.analysis_service import AnalysisService
from ..services.sentiment_service import SentimentService

router = APIRouter()
analysis_service = AnalysisService()
sentiment_service = SentimentService()

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "1y"
    analysis_type: str = "comprehensive"  # comprehensive, technical, fundamental, sentiment

@router.post("/comprehensive")
async def comprehensive_analysis(request: AnalysisRequest):
    """Perform comprehensive financial analysis including technical, fundamental, and sentiment analysis"""
    try:
        analysis = await analysis_service.comprehensive_analysis(
            request.symbol, 
            request.timeframe, 
            request.analysis_type
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/technical/{symbol}")
async def technical_analysis(
    symbol: str,
    timeframe: str = Query("1y", description="Analysis timeframe"),
    indicators: str = Query("all", description="Technical indicators to include")
):
    """Perform technical analysis with various indicators"""
    try:
        analysis = await analysis_service.technical_analysis(symbol, timeframe, indicators)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/fundamental/{symbol}")
async def fundamental_analysis(symbol: str):
    """Perform fundamental analysis"""
    try:
        analysis = await analysis_service.fundamental_analysis(symbol)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sentiment/{symbol}")
async def sentiment_analysis(symbol: str):
    """Perform sentiment analysis on news and social media"""
    try:
        sentiment = await sentiment_service.analyze_sentiment(symbol)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/risk/{symbol}")
async def risk_analysis(symbol: str):
    """Perform risk assessment analysis"""
    try:
        risk = await analysis_service.risk_analysis(symbol)
        return risk
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/portfolio/optimization")
async def portfolio_optimization(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    risk_tolerance: str = Query("medium", description="Risk tolerance: low, medium, high")
):
    """Optimize portfolio allocation"""
    try:
        symbols_list = [s.strip() for s in symbols.split(",")]
        optimization = await analysis_service.portfolio_optimization(symbols_list, risk_tolerance)
        return optimization
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
