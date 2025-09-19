from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from ..services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "1y"
    prediction_days: int = 30
    model_type: str = "lstm"  # lstm, arima, linear_regression

@router.post("/price")
async def predict_price(request: PredictionRequest):
    """Predict future stock prices using AI models"""
    try:
        prediction = await prediction_service.predict_price(
            request.symbol,
            request.timeframe,
            request.prediction_days,
            request.model_type
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/trend/{symbol}")
async def predict_trend(
    symbol: str,
    timeframe: str = Query("1y", description="Analysis timeframe"),
    prediction_days: int = Query(30, description="Days to predict ahead")
):
    """Predict market trend direction"""
    try:
        trend = await prediction_service.predict_trend(symbol, timeframe, prediction_days)
        return trend
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/volatility/{symbol}")
async def predict_volatility(
    symbol: str,
    timeframe: str = Query("1y", description="Analysis timeframe")
):
    """Predict future volatility"""
    try:
        volatility = await prediction_service.predict_volatility(symbol, timeframe)
        return volatility
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/portfolio/returns")
async def predict_portfolio_returns(
    symbols: List[str],
    weights: Optional[List[float]] = None,
    prediction_days: int = 30
):
    """Predict portfolio returns"""
    try:
        if len(symbols) != len(weights) if weights else False:
            raise HTTPException(status_code=400, detail="Number of symbols and weights must match")
        
        returns = await prediction_service.predict_portfolio_returns(symbols, weights, prediction_days)
        return returns
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/market/crash-probability")
async def predict_market_crash_probability():
    """Predict probability of market crash using multiple indicators"""
    try:
        probability = await prediction_service.predict_market_crash_probability()
        return probability
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
