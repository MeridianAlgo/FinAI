import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  Globe,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  Clock,
  DollarSign,
  Zap,
  RefreshCw
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { toast } from 'react-hot-toast';
import api from '../services/api';

const Market = () => {
  const [marketData, setMarketData] = useState(null);
  const [trendingStocks, setTrendingStocks] = useState([]);
  const [crashPrediction, setCrashPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchMarketData();
  }, []);

  const fetchMarketData = async () => {
    try {
      setLoading(true);
      const [marketResponse, trendingResponse, crashResponse] = await Promise.all([
        api.get('/api/data/market/indices'),
        api.get('/api/data/market/trending'),
        api.get('/api/predictions/market/crash-probability')
      ]);
      
      setMarketData(marketResponse.data);
      setTrendingStocks(trendingResponse.data);
      setCrashPrediction(crashResponse.data);
    } catch (error) {
      toast.error('Failed to fetch market data');
      console.error('Market data error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchMarketData();
    setRefreshing(false);
    toast.success('Market data refreshed');
  };

  const getChangeColor = (change) => {
    if (change > 0) return 'text-success-600';
    if (change < 0) return 'text-danger-600';
    return 'text-gray-600';
  };

  const getChangeIcon = (change) => {
    if (change > 0) return <TrendingUp className="w-4 h-4" />;
    if (change < 0) return <TrendingDown className="w-4 h-4" />;
    return <Activity className="w-4 h-4" />;
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'very_low':
        return 'text-success-600';
      case 'low':
        return 'text-success-500';
      case 'medium':
        return 'text-warning-600';
      case 'high':
        return 'text-danger-500';
      case 'very_high':
        return 'text-danger-600';
      default:
        return 'text-gray-600';
    }
  };

  const getRiskBgColor = (risk) => {
    switch (risk) {
      case 'very_low':
        return 'bg-success-100';
      case 'low':
        return 'bg-success-50';
      case 'medium':
        return 'bg-warning-50';
      case 'high':
        return 'bg-danger-50';
      case 'very_high':
        return 'bg-danger-100';
      default:
        return 'bg-gray-50';
    }
  };

  const formatNumber = (num) => {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(2);
  };

  const prepareMarketChartData = () => {
    if (!marketData) return [];
    
    return Object.entries(marketData).map(([name, data]) => ({
      name: name,
      value: data.price,
      change: data.change_percent
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Market Overview</h1>
            <p className="text-gray-600">Real-time market data and AI-powered insights</p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="btn btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </motion.div>

      {/* Market Indices */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
      >
        {marketData && Object.entries(marketData).map(([name, data], index) => (
          <div key={name} className="metric-card hover-lift">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600">{name}</h3>
              <div className={`flex items-center space-x-1 ${getChangeColor(data.change_percent)}`}>
                {getChangeIcon(data.change_percent)}
                <span className="text-sm font-medium">
                  {data.change_percent > 0 ? '+' : ''}{data.change_percent.toFixed(2)}%
                </span>
              </div>
            </div>
            <div className="text-2xl font-bold text-gray-900">
              {data.symbol === '^VIX' ? data.price.toFixed(2) : formatNumber(data.price)}
            </div>
            <div className="text-sm text-gray-500 mt-1">
              {data.change > 0 ? '+' : ''}{data.change.toFixed(2)}
            </div>
          </div>
        ))}
      </motion.div>

      {/* Market Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Market Performance Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Market Performance</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={prepareMarketChartData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                  formatter={(value, name) => [
                    name === 'value' ? `$${value.toFixed(2)}` : `${value.toFixed(2)}%`,
                    name === 'value' ? 'Price' : 'Change'
                  ]}
                />
                <Bar 
                  dataKey="value" 
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Trending Stocks */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Trending Stocks</h2>
          <div className="space-y-4">
            {trendingStocks.slice(0, 6).map((stock, index) => (
              <div key={stock.symbol} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                    <span className="text-sm font-medium text-primary-700">
                      {stock.symbol.charAt(0)}
                    </span>
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">{stock.symbol}</div>
                    <div className="text-sm text-gray-600">{stock.name}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium text-gray-900">
                    ${stock.price.toFixed(2)}
                  </div>
                  <div className={`text-sm flex items-center space-x-1 ${getChangeColor(stock.change_percent)}`}>
                    {getChangeIcon(stock.change_percent)}
                    <span>{stock.change_percent > 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Market Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Market Sentiment */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-success-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-success-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">Market Sentiment</h3>
          </div>
          <div className="text-3xl font-bold text-success-600 mb-2">Bullish</div>
          <p className="text-gray-600 text-sm">
            Overall market sentiment is positive with strong fundamentals and investor confidence.
          </p>
        </motion.div>

        {/* Volatility Index */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-warning-100 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-warning-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">Volatility (VIX)</h3>
          </div>
          <div className="text-3xl font-bold text-warning-600 mb-2">
            {marketData?.['VIX']?.price?.toFixed(2) || 'N/A'}
          </div>
          <p className="text-gray-600 text-sm">
            {marketData?.['VIX']?.price > 30 ? 'High volatility - market stress' :
             marketData?.['VIX']?.price > 20 ? 'Moderate volatility' : 'Low volatility - calm market'}
          </p>
        </motion.div>

        {/* Market Trend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card"
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-primary-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">Market Trend</h3>
          </div>
          <div className="text-3xl font-bold text-primary-600 mb-2">Uptrend</div>
          <p className="text-gray-600 text-sm">
            Major indices showing positive momentum with strong technical indicators.
          </p>
        </motion.div>
      </div>

      {/* Crash Prediction */}
      {crashPrediction && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center space-x-2">
            <AlertTriangle className="w-6 h-6 text-warning-600" />
            <span>Market Crash Probability</span>
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            <div className="text-center">
              <div className={`w-20 h-20 mx-auto rounded-full ${getRiskBgColor(crashPrediction.risk_level)} flex items-center justify-center mb-3`}>
                <span className={`text-2xl font-bold ${getRiskColor(crashPrediction.risk_level)}`}>
                  {crashPrediction.crash_probability?.toFixed(1) || '0.0'}%
                </span>
              </div>
              <h3 className="font-semibold text-gray-900">Crash Probability</h3>
              <p className="text-sm text-gray-600">Next 30 days</p>
            </div>
            
            <div className="text-center">
              <div className={`w-20 h-20 mx-auto rounded-full ${getRiskBgColor(crashPrediction.risk_level)} flex items-center justify-center mb-3`}>
                <span className={`text-2xl font-bold ${getRiskColor(crashPrediction.risk_level)}`}>
                  {crashPrediction.risk_level || 'medium'}
                </span>
              </div>
              <h3 className="font-semibold text-gray-900">Risk Level</h3>
              <p className="text-sm text-gray-600">Overall assessment</p>
            </div>
            
            <div className="text-center">
              <div className="w-20 h-20 mx-auto rounded-full bg-blue-100 flex items-center justify-center mb-3">
                <span className="text-2xl font-bold text-blue-600">
                  {crashPrediction.indicators?.vix_signal ? 'High' : 'Low'}
                </span>
              </div>
              <h3 className="font-semibold text-gray-900">VIX Signal</h3>
              <p className="text-sm text-gray-600">Volatility indicator</p>
            </div>
            
            <div className="text-center">
              <div className="w-20 h-20 mx-auto rounded-full bg-purple-100 flex items-center justify-center mb-3">
                <span className="text-2xl font-bold text-purple-600">
                  {crashPrediction.indicators?.rate_signal ? 'High' : 'Low'}
                </span>
              </div>
              <h3 className="font-semibold text-gray-900">Rate Signal</h3>
              <p className="text-sm text-gray-600">Interest rate impact</p>
            </div>
          </div>
          
          {/* Market Conditions */}
          {crashPrediction.market_conditions && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Market Conditions</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Volatility</span>
                  <span className={`status-indicator ${
                    crashPrediction.market_conditions.volatility === 'high' ? 'status-warning' :
                    crashPrediction.market_conditions.volatility === 'low' ? 'status-success' : 'status-info'
                  }`}>
                    {crashPrediction.market_conditions.volatility}
                  </span>
                </div>
                <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Interest Rates</span>
                  <span className={`status-indicator ${
                    crashPrediction.market_conditions.interest_rates === 'high' ? 'status-warning' :
                    crashPrediction.market_conditions.interest_rates === 'low' ? 'status-success' : 'status-info'
                  }`}>
                    {crashPrediction.market_conditions.interest_rates}
                  </span>
                </div>
              </div>
            </div>
          )}
          
          {/* Recommendations */}
          {crashPrediction.recommendations && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Recommendations</h3>
              <div className="space-y-3">
                {crashPrediction.recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
                    <Zap className="w-5 h-5 text-blue-600 mt-0.5" />
                    <span className="text-sm text-blue-800">{recommendation}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default Market;
