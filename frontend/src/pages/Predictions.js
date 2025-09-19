import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  TrendingUp, 
  TrendingDown, 
  Brain,
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  Activity,
  Zap
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { toast } from 'react-hot-toast';
import api from '../services/api';

const Predictions = () => {
  const [searchSymbol, setSearchSymbol] = useState('');
  const [predictionData, setPredictionData] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [volatilityData, setVolatilityData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('lstm');
  const [predictionDays, setPredictionDays] = useState(30);

  const models = [
    { id: 'lstm', name: 'LSTM Neural Network', description: 'Deep learning model for time series prediction' },
    { id: 'arima', name: 'ARIMA', description: 'Statistical model for trend analysis' },
    { id: 'linear_regression', name: 'Linear Regression', description: 'Machine learning model for price prediction' },
    { id: 'ensemble', name: 'Ensemble Model', description: 'Combined approach using multiple algorithms' }
  ];

  const handlePrediction = async () => {
    if (!searchSymbol.trim()) {
      toast.error('Please enter a stock symbol');
      return;
    }

    try {
      setLoading(true);
      const [priceResponse, trendResponse, volatilityResponse] = await Promise.all([
        api.post('/api/predictions/price', {
          symbol: searchSymbol.toUpperCase(),
          timeframe: '1y',
          prediction_days: predictionDays,
          model_type: selectedModel
        }),
        api.get(`/api/predictions/trend/${searchSymbol.toUpperCase()}?timeframe=1y&prediction_days=${predictionDays}`),
        api.get(`/api/predictions/volatility/${searchSymbol.toUpperCase()}?timeframe=1y`)
      ]);
      
      setPredictionData(priceResponse.data);
      setTrendData(trendResponse.data);
      setVolatilityData(volatilityResponse.data);
      toast.success('Predictions generated successfully');
    } catch (error) {
      toast.error(error.message || 'Failed to generate predictions');
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getTrendColor = (trend) => {
    switch (trend) {
      case 'strong_uptrend':
      case 'uptrend':
        return 'text-success-600';
      case 'strong_downtrend':
      case 'downtrend':
        return 'text-danger-600';
      default:
        return 'text-gray-600';
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'strong_uptrend':
      case 'uptrend':
        return <TrendingUp className="w-5 h-5" />;
      case 'strong_downtrend':
      case 'downtrend':
        return <TrendingDown className="w-5 h-5" />;
      default:
        return <Activity className="w-5 h-5" />;
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'low':
        return 'text-success-600';
      case 'medium':
        return 'text-warning-600';
      case 'high':
        return 'text-danger-600';
      default:
        return 'text-gray-600';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString();
  };

  const prepareChartData = (predictions) => {
    if (!predictions) return [];
    
    return predictions.map((pred, index) => ({
      day: index + 1,
      predicted: pred.predicted_price,
      lower: pred.confidence_lower,
      upper: pred.confidence_upper,
      date: pred.date
    }));
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Predictions</h1>
        <p className="text-gray-600">Machine learning-powered price and trend predictions</p>
      </motion.div>

      {/* Search Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card mb-8"
      >
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="md:col-span-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
                value={searchSymbol}
                onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                className="input pl-10"
                onKeyPress={(e) => e.key === 'Enter' && handlePrediction()}
              />
            </div>
          </div>
          
          <div>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="input"
            >
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <select
              value={predictionDays}
              onChange={(e) => setPredictionDays(parseInt(e.target.value))}
              className="input"
            >
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
              <option value={60}>60 days</option>
              <option value={90}>90 days</option>
            </select>
          </div>
        </div>
        
        <div className="mt-4 flex justify-between items-center">
          <div className="text-sm text-gray-600">
            {models.find(m => m.id === selectedModel)?.description}
          </div>
          <button
            onClick={handlePrediction}
            disabled={loading}
            className="btn btn-primary flex items-center space-x-2"
          >
            {loading ? (
              <div className="spinner w-4 h-4"></div>
            ) : (
              <Brain className="w-4 h-4" />
            )}
            <span>Generate Predictions</span>
          </button>
        </div>
      </motion.div>

      {/* Prediction Results */}
      {predictionData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-8"
        >
          {/* Price Prediction Chart */}
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Price Prediction for {predictionData.symbol}
              </h2>
              <div className="flex items-center space-x-4">
                <div className="text-sm text-gray-600">
                  Model: {models.find(m => m.id === predictionData.model_type)?.name}
                </div>
                <div className="text-sm text-gray-600">
                  Accuracy: {predictionData.prediction_accuracy?.toFixed(1)}%
                </div>
              </div>
            </div>
            
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={prepareChartData(predictionData.predictions)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="day" 
                    stroke="#6b7280"
                    label={{ value: 'Days Ahead', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    stroke="#6b7280"
                    label={{ value: 'Price ($)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                    formatter={(value, name) => [
                      `$${value.toFixed(2)}`,
                      name === 'predicted' ? 'Predicted Price' : 
                      name === 'lower' ? 'Lower Bound' : 'Upper Bound'
                    ]}
                    labelFormatter={(day) => `Day ${day}`}
                  />
                  <Area
                    type="monotone"
                    dataKey="upper"
                    stackId="1"
                    stroke="none"
                    fill="#3b82f6"
                    fillOpacity={0.1}
                  />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    stackId="1"
                    stroke="none"
                    fill="#fff"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#3b82f6"
                    strokeWidth={3}
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="metric-card">
                <h3 className="font-medium text-gray-900 mb-2">Current Price</h3>
                <div className="text-2xl font-bold text-gray-900">
                  ${predictionData.current_price?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="metric-card">
                <h3 className="font-medium text-gray-900 mb-2">Predicted Price</h3>
                <div className="text-2xl font-bold text-gray-900">
                  ${predictionData.predictions?.[predictionData.predictions.length - 1]?.predicted_price?.toFixed(2) || 'N/A'}
                </div>
              </div>
              
              <div className="metric-card">
                <h3 className="font-medium text-gray-900 mb-2">Expected Change</h3>
                <div className={`text-2xl font-bold flex items-center space-x-2 ${
                  predictionData.trend_direction?.includes('up') ? 'text-success-600' : 
                  predictionData.trend_direction?.includes('down') ? 'text-danger-600' : 'text-gray-600'
                }`}>
                  {getTrendIcon(predictionData.trend_direction)}
                  <span>{predictionData.trend_direction || 'neutral'}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Trend Analysis */}
          {trendData && (
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <TrendingUp className="w-5 h-5 text-primary-600" />
                <span>Trend Analysis</span>
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Trend Direction</h4>
                  <div className={`text-2xl font-bold flex items-center space-x-2 ${getTrendColor(trendData.trend_direction)}`}>
                    {getTrendIcon(trendData.trend_direction)}
                    <span className="capitalize">{trendData.trend_direction?.replace('_', ' ') || 'neutral'}</span>
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Trend Strength</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {trendData.trend_strength?.toFixed(1) || '0.0'}%
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Confidence</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {(trendData.confidence * 100)?.toFixed(1) || '0.0'}%
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Risk Level</h4>
                  <div className={`text-2xl font-bold ${getRiskColor(predictionData.risk_assessment)}`}>
                    {predictionData.risk_assessment || 'medium'}
                  </div>
                </div>
              </div>
              
              {/* Support and Resistance */}
              {trendData.support_resistance && (
                <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Support & Resistance</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600">Support Level</span>
                        <span className="font-medium">
                          ${trendData.support_resistance.support?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600">Resistance Level</span>
                        <span className="font-medium">
                          ${trendData.support_resistance.resistance?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600">Current Price</span>
                        <span className="font-medium">
                          ${trendData.support_resistance.current_price?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium text-gray-900 mb-3">Volatility Forecast</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600">Current Volatility</span>
                        <span className="font-medium">
                          {(trendData.volatility_forecast?.current_volatility * 100)?.toFixed(1) || '0.0'}%
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600">Trend</span>
                        <span className={`status-indicator ${
                          trendData.volatility_forecast?.trend === 'increasing' ? 'status-warning' :
                          trendData.volatility_forecast?.trend === 'decreasing' ? 'status-success' : 'status-info'
                        }`}>
                          {trendData.volatility_forecast?.trend || 'stable'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Volatility Analysis */}
          {volatilityData && (
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <Activity className="w-5 h-5 text-primary-600" />
                <span>Volatility Analysis</span>
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Current Volatility</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {(volatilityData.current_volatility * 100)?.toFixed(1) || '0.0'}%
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Predicted Volatility</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {(volatilityData.predicted_volatility * 100)?.toFixed(1) || '0.0'}%
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Risk Level</h4>
                  <div className={`text-2xl font-bold ${getRiskColor(volatilityData.risk_level)}`}>
                    {volatilityData.risk_level || 'medium'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Volatility Trend</h4>
                  <div className={`text-2xl font-bold flex items-center space-x-2 ${
                    volatilityData.volatility_trend === 'increasing' ? 'text-warning-600' :
                    volatilityData.volatility_trend === 'decreasing' ? 'text-success-600' : 'text-gray-600'
                  }`}>
                    {volatilityData.volatility_trend === 'increasing' ? (
                      <TrendingUp className="w-5 h-5" />
                    ) : volatilityData.volatility_trend === 'decreasing' ? (
                      <TrendingDown className="w-5 h-5" />
                    ) : (
                      <Activity className="w-5 h-5" />
                    )}
                    <span className="capitalize">{volatilityData.volatility_trend || 'stable'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* AI Insights */}
          <div className="card">
            <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
              <Zap className="w-5 h-5 text-primary-600" />
              <span>AI Insights & Recommendations</span>
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
                  <CheckCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-blue-900">Model Performance</h4>
                    <p className="text-sm text-blue-700">
                      The {models.find(m => m.id === predictionData.model_type)?.name} model shows 
                      {predictionData.prediction_accuracy > 70 ? ' strong' : 
                       predictionData.prediction_accuracy > 50 ? ' moderate' : ' weak'} 
                      predictive accuracy of {predictionData.prediction_accuracy?.toFixed(1)}%.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3 p-4 bg-green-50 rounded-lg">
                  <Target className="w-5 h-5 text-green-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-green-900">Investment Outlook</h4>
                    <p className="text-sm text-green-700">
                      Based on the {predictionData.trend_direction || 'neutral'} trend, 
                      the stock is expected to {predictionData.trend_direction?.includes('up') ? 'appreciate' : 
                      predictionData.trend_direction?.includes('down') ? 'depreciate' : 'remain stable'} 
                      over the next {predictionDays} days.
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-start space-x-3 p-4 bg-yellow-50 rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-yellow-900">Risk Assessment</h4>
                    <p className="text-sm text-yellow-700">
                      The {predictionData.risk_assessment || 'medium'} risk level suggests 
                      {predictionData.risk_assessment === 'high' ? ' caution and proper risk management' :
                       predictionData.risk_assessment === 'low' ? ' relatively stable conditions' :
                       ' moderate volatility expectations'} for this investment.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3 p-4 bg-purple-50 rounded-lg">
                  <Clock className="w-5 h-5 text-purple-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-purple-900">Time Horizon</h4>
                    <p className="text-sm text-purple-700">
                      Predictions are most reliable for the {predictionDays}-day timeframe. 
                      Longer-term predictions may require additional analysis and monitoring.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Predictions;
