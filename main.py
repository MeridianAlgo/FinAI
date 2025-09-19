from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from config import settings
from api.routes import analysis, data, predictions
import os

app = FastAPI(
    title="MeridianAI Financial Analysis",
    description="AI-powered financial analysis and prediction platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])

# Include Enhanced AI routes
from api.routes import enhanced_analysis
app.include_router(enhanced_analysis.router, prefix="/api/ai", tags=["enhanced-ai"])

# Serve static files
if os.path.exists("frontend/build"):
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse("frontend/build/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "MeridianAI Financial Analysis API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
