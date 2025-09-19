from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
from ..services.enhanced_ai_service import EnhancedAIService
import asyncio

router = APIRouter()
ai_service = EnhancedAIService()

class AnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"  # comprehensive, price, trend, risk, sentiment, fundamental

@router.on_event("startup")
async def startup_event():
    """Initialize and train AI models on startup"""
    print("🚀 Initializing Enhanced AI Service...")
    await ai_service.train_ai_models()

@router.post("/company/{symbol}")
async def analyze_company(symbol: str, request: AnalysisRequest):
    """Comprehensive AI-powered company analysis"""
    try:
        analysis = await ai_service.analyze_company(symbol.upper())
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/ai-score/{symbol}")
async def get_ai_score(symbol: str):
    """Get AI-generated investment score for a company"""
    try:
        analysis = await ai_service.analyze_company(symbol.upper())
        return {
            "symbol": symbol.upper(),
            "ai_score": analysis["ai_score"],
            "recommendation": analysis["recommendation"],
            "confidence": analysis["confidence"],
            "key_insights": analysis["key_insights"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/price-prediction/{symbol}")
async def get_price_prediction(symbol: str, days: int = Query(30, description="Days to predict ahead")):
    """Get AI-powered price prediction"""
    try:
        analysis = await ai_service.analyze_company(symbol.upper())
        price_prediction = analysis["price_prediction"]
        
        # Generate multiple day predictions
        current_price = price_prediction["current_price"]
        expected_return = price_prediction["expected_return"]
        
        predictions = []
        for day in range(1, days + 1):
            predicted_price = current_price * (1 + expected_return * (day / 30))
            predictions.append({
                "day": day,
                "predicted_price": predicted_price,
                "confidence": price_prediction["confidence"] * (1 - day / 100)  # Decreasing confidence over time
            })
        
        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "predictions": predictions,
            "direction": price_prediction["direction"],
            "confidence": price_prediction["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/trend-analysis/{symbol}")
async def get_trend_analysis(symbol: str):
    """Get AI-powered trend analysis"""
    try:
        analysis = await ai_service.analyze_company(symbol.upper())
        trend_analysis = analysis["trend_analysis"]
        
        return {
            "symbol": symbol.upper(),
            "trend_direction": trend_analysis["direction"],
            "trend_strength": trend_analysis["strength"],
            "probability": trend_analysis["probability"],
            "confidence": trend_analysis["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/risk-assessment/{symbol}")
async def get_risk_assessment(symbol: str):
    """Get AI-powered risk assessment"""
    try:
        analysis = await ai_service.analyze_company(symbol.upper())
        risk_assessment = analysis["risk_assessment"]
        
        return {
            "symbol": symbol.upper(),
            "risk_level": risk_assessment["risk_level"],
            "risk_score": risk_assessment["risk_score"],
            "predicted_volatility": risk_assessment["predicted_volatility"],
            "confidence": risk_assessment["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sentiment-analysis/{symbol}")
async def get_sentiment_analysis(symbol: str):
    """Get AI-powered sentiment analysis"""
    try:
        analysis = await ai_service.analyze_company(symbol.upper())
        sentiment_analysis = analysis["sentiment_analysis"]
        
        return {
            "symbol": symbol.upper(),
            "sentiment": sentiment_analysis["sentiment"],
            "sentiment_score": sentiment_analysis["score"],
            "confidence": sentiment_analysis["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/fundamental-score/{symbol}")
async def get_fundamental_score(symbol: str):
    """Get AI-powered fundamental analysis score"""
    try:
        analysis = await ai_service.analyze_company(symbol.upper())
        fundamental_score = analysis["fundamental_score"]
        
        return {
            "symbol": symbol.upper(),
            "fundamental_score": fundamental_score["score"],
            "grade": fundamental_score["grade"],
            "expected_return": fundamental_score["expected_return"],
            "confidence": fundamental_score["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare-companies")
async def compare_companies(symbols: List[str]):
    """Compare multiple companies using AI analysis"""
    try:
        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 companies allowed for comparison")
        
        comparisons = []
        for symbol in symbols:
            try:
                analysis = await ai_service.analyze_company(symbol.upper())
                comparisons.append({
                    "symbol": symbol.upper(),
                    "ai_score": analysis["ai_score"],
                    "recommendation": analysis["recommendation"],
                    "confidence": analysis["confidence"],
                    "price_prediction": analysis["price_prediction"]["direction"],
                    "trend_analysis": analysis["trend_analysis"]["direction"],
                    "risk_level": analysis["risk_assessment"]["risk_level"],
                    "sentiment": analysis["sentiment_analysis"]["sentiment"],
                    "fundamental_grade": analysis["fundamental_score"]["grade"]
                })
            except Exception as e:
                comparisons.append({
                    "symbol": symbol.upper(),
                    "error": str(e)
                })
        
        # Sort by AI score
        comparisons.sort(key=lambda x: x.get("ai_score", 0), reverse=True)
        
        return {
            "comparisons": comparisons,
            "top_pick": comparisons[0] if comparisons else None,
            "analysis_date": analysis["analysis_date"] if comparisons else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/market-insights")
async def get_market_insights():
    """Get AI-generated market insights"""
    try:
        # Analyze major market indices
        major_indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
        market_analysis = []
        
        for index in major_indices:
            try:
                analysis = await ai_service.analyze_company(index)
                market_analysis.append({
                    "index": index,
                    "ai_score": analysis["ai_score"],
                    "trend": analysis["trend_analysis"]["direction"],
                    "risk": analysis["risk_assessment"]["risk_level"],
                    "sentiment": analysis["sentiment_analysis"]["sentiment"]
                })
            except Exception:
                continue
        
        # Generate overall market insights
        avg_score = sum(ma["ai_score"] for ma in market_analysis) / len(market_analysis) if market_analysis else 50
        bullish_count = sum(1 for ma in market_analysis if "up" in ma["trend"])
        risk_levels = [ma["risk"] for ma in market_analysis]
        
        return {
            "overall_market_score": avg_score,
            "market_sentiment": "bullish" if bullish_count > len(market_analysis) / 2 else "bearish",
            "dominant_risk_level": max(set(risk_levels), key=risk_levels.count) if risk_levels else "medium",
            "index_analysis": market_analysis,
            "insights": [
                f"Overall market AI score: {avg_score:.1f}/100",
                f"Market trend: {'Bullish' if bullish_count > len(market_analysis) / 2 else 'Bearish'}",
                f"Dominant risk level: {max(set(risk_levels), key=risk_levels.count) if risk_levels else 'medium'}",
                "Analysis based on comprehensive AI models trained on market data"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/retrain-models")
async def retrain_models(background_tasks: BackgroundTasks):
    """Retrain AI models with latest data"""
    try:
        background_tasks.add_task(ai_service.train_ai_models)
        return {"message": "AI models retraining started in background"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model-status")
async def get_model_status():
    """Get status of AI models"""
    try:
        return {
            "models_trained": len(ai_service.trained_models),
            "available_models": list(ai_service.models.keys()),
            "trained_features": list(ai_service.trained_models.keys()),
            "status": "ready" if ai_service.trained_models else "training"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
