from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from ..services.data_service import DataService

router = APIRouter()
data_service = DataService()

@router.get("/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    period: str = Query("1y", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"),
    interval: str = Query("1d", description="Data interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
):
    """Get historical stock data for a given symbol"""
    try:
        data = await data_service.get_stock_data(symbol, period, interval)
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "data": data.to_dict('records'),
            "metadata": {
                "total_records": len(data),
                "date_range": {
                    "start": data.index[0].isoformat() if len(data) > 0 else None,
                    "end": data.index[-1].isoformat() if len(data) > 0 else None
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str):
    """Get detailed information about a stock"""
    try:
        info = await data_service.get_stock_info(symbol)
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/market/trending")
async def get_trending_stocks():
    """Get trending stocks from major indices"""
    try:
        trending = await data_service.get_trending_stocks()
        return trending
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/market/indices")
async def get_market_indices():
    """Get major market indices data"""
    try:
        indices = await data_service.get_market_indices()
        return indices
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare")
async def compare_stocks(symbols: List[str]):
    """Compare multiple stocks"""
    try:
        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
        
        comparison = await data_service.compare_stocks(symbols)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
