# MeridianAI Financial Analysis CLI

A minimalistic command-line financial analysis tool with advanced AI models that provides accurate price predictions, trend analysis, and risk assessment using 10 years of historical data.

## Features

- **Lazy Loading**: Models train only when needed for specific analysis
- **10-Year Data**: Uses extended historical data for improved accuracy
- **Advanced ML**: Sophisticated algorithms with enhanced parameters
- **Minimalistic UI**: Clean command-line interface with loading indicators
- **No API Keys**: All data from public sources (Yahoo Finance)
- **Real-time Analysis**: Live market data and predictions

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MeridianAI

# Install dependencies
pip install -r requirements.txt

# Run the CLI
python cli_interface.py
```

### Basic Usage

```bash
# Start the CLI
python cli_interface.py

# Follow the menu prompts
# 1. Analyze Company - Enter symbol (e.g., AAPL)
# 2. Compare Companies - Enter multiple symbols
# 3. Price Prediction - Get detailed price forecasts
# 4. Trend Analysis - Analyze trend direction and strength
# 5. Risk Assessment - Evaluate investment risk
# 6. Market Overview - Get market-wide analysis
# 7. Train Models - Advanced training with comprehensive data
```

## AI Models

### Price Prediction Model
- **Algorithm**: Random Forest Regressor (200 estimators, max depth 15)
- **Features**: 13 technical and fundamental indicators
- **Data**: 10 years of historical price data
- **Output**: 30-day price forecasts with confidence intervals

### Trend Analysis Model
- **Algorithm**: Gradient Boosting Regressor (150 estimators)
- **Features**: 9 trend-related indicators
- **Data**: Extended historical trend patterns
- **Output**: Trend direction, strength, and probability

### Risk Assessment Model
- **Algorithm**: Random Forest Regressor (100 estimators)
- **Features**: 7 risk-related indicators
- **Data**: Volatility and risk patterns
- **Output**: Risk level, score, and predicted volatility

### Sentiment Analysis Model
- **Algorithm**: Linear Regression with feature scaling
- **Features**: 6 sentiment indicators
- **Data**: Price action and volume patterns
- **Output**: Sentiment score and classification

### Fundamental Scoring Model
- **Algorithm**: Ridge Regression
- **Features**: 4 key fundamental ratios
- **Data**: Financial metrics and performance
- **Output**: Fundamental score and letter grade

## Technical Indicators

### Moving Averages
- Simple Moving Average (20, 50, 200 days)
- Exponential Moving Average (12, 26 days)

### Momentum Indicators
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R

### Volatility Indicators
- Bollinger Bands
- Average True Range (ATR)
- Historical Volatility (10, 20, 30 days)

### Volume Indicators
- Volume Ratio
- Volume Moving Average

### Price Action
- Price Momentum (1, 5, 10, 20 days)
- Price Change Percentages
- Support and Resistance Levels

## Data Sources

- **Yahoo Finance**: Historical price data, company information
- **Public APIs**: No authentication required
- **Real-time Data**: Live market prices and volumes
- **Extended History**: 10 years of data for improved accuracy

## Performance

### Training Time
- **Single Company**: 30-60 seconds (10 years of data)
- **Multiple Companies**: 2-5 minutes per company
- **Comprehensive Training**: 5-10 minutes (100+ stocks)

### Analysis Speed
- **Company Analysis**: 5-10 seconds
- **Price Prediction**: 3-5 seconds
- **Trend Analysis**: 2-3 seconds
- **Risk Assessment**: 2-3 seconds

### Accuracy Improvements
- **Extended Data**: 10 years vs 2 years (5x more data)
- **Advanced Models**: Enhanced parameters and algorithms
- **Feature Engineering**: 20+ technical indicators
- **Model Optimization**: Cross-validation and hyperparameter tuning

## Example Output

### Company Analysis
```
Analysis Results for AAPL
========================================
AI Score: 78.5/100
Recommendation: Buy
Confidence: 85.2%

Price Prediction:
  Current: $237.88
  Predicted: $245.32
  Expected Return: 3.13%
  Direction: bullish

Trend Analysis:
  Direction: uptrend
  Strength: 72.5%
  Probability: 68.3%

Risk Assessment:
  Level: medium
  Score: 35.2/100
  Volatility: 18.45%

Sentiment Analysis:
  Sentiment: positive
  Score: 78.3/100

Fundamental Analysis:
  Score: 82.1/100
  Grade: A
  Expected Return: 4.2%
```

### Price Prediction
```
Price Prediction for AAPL
===================================
Current Price: $237.88
Predicted Price: $245.32
Expected Return: 3.13%
Direction: bullish
Confidence: 85.2%

30-Day Price Forecast:
-------------------------
Day  7: $ 239.45 (+0.66%)
Day 14: $ 241.02 (+1.32%)
Day 21: $ 242.59 (+1.98%)
Day 30: $ 245.32 (+3.13%)
```

## Architecture

### Core Components
- `cli_interface.py`: Main CLI application
- `api/services/enhanced_ai_service.py`: AI model service
- `api/services/data_training_service.py`: Data processing service
- `main.py`: FastAPI backend server

### Data Flow
1. **User Input**: Symbol selection via CLI
2. **Data Gathering**: 10 years of historical data from Yahoo Finance
3. **Feature Engineering**: Calculate 20+ technical indicators
4. **Model Training**: Train 5 specialized AI models
5. **Analysis**: Generate predictions and insights
6. **Output**: Display results in clean format

### Model Architecture
- **Input Layer**: 13-20 features per model
- **Processing Layer**: Scaled and normalized features
- **Model Layer**: Specialized algorithms per task
- **Output Layer**: Predictions with confidence scores

## Configuration

### Model Parameters
```python
# Price Prediction Model
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Trend Analysis Model
GradientBoostingRegressor(
    n_estimators=150,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
```

### Data Parameters
- **Historical Period**: 10 years
- **Training Samples**: 200+ per company
- **Feature Count**: 13-20 per model
- **Prediction Horizon**: 30 days

## Error Handling

- **Data Validation**: Check for valid symbols and data availability
- **Model Fallbacks**: Default values for missing features
- **Graceful Degradation**: Continue operation with partial data
- **User Feedback**: Clear error messages and suggestions

## Dependencies

```
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
yfinance==0.2.18
requests==2.31.0
python-multipart==0.0.6
pydantic==2.5.0
plotly==5.17.0
ta==0.10.2
beautifulsoup4==4.12.2
aiohttp==3.9.1
scipy==1.11.4
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check the documentation
2. Review error messages
3. Test with different symbols
4. Check data availability

## Changelog

### Version 2.0
- Lazy loading implementation
- 10-year data support
- Enhanced ML models
- Minimalistic CLI interface
- Improved accuracy and performance

### Version 1.0
- Initial release
- Basic AI models
- Terminal interface
- Yahoo Finance integration
