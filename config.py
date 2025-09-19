import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./financial_analysis.db")
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    
    # Financial Data Settings
    DEFAULT_TIMEFRAME = "1y"
    MAX_SYMBOLS = 10

settings = Settings()
