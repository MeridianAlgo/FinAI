import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { toast } from 'react-hot-toast';
import api from '../services/api';

const Dashboard = () => {
  const [marketData, setMarketData] = useState(null);
  const [trendingStocks, setTrendingStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1d');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [marketResponse, trendingResponse] = await Promise.all([
        api.get('/api/data/market/indices'),
        api.get('/api/data/market/trending')
      ]);
      
      setMarketData(marketResponse.data);
      setTrendingStocks(trendingResponse.data);
    } catch (error) {
      toast.error('Failed to fetch dashboard data');
      console.error('Dashboard data error:', error);
    } finally {
      setLoading(false);
    }
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

  const formatNumber = (num) => {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(2);
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
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Financial Dashboard</h1>
        <p className="text-gray-600">Real-time market insights and AI-powered analysis</p>
      </motion.div>

      {/* Market Overview */}
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

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Market Performance Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Market Performance</h2>
            <div className="flex space-x-2">
              {['1d', '1w', '1m', '3m'].map((timeframe) => (
                <button
                  key={timeframe}
                  onClick={() => setSelectedTimeframe(timeframe)}
                  className={`px-3 py-1 text-sm rounded-lg transition-colors duration-200 ${
                    selectedTimeframe === timeframe
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {timeframe}
                </button>
              ))}
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={[]}>
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
                />
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#3b82f6" 
                  fill="url(#colorGradient)" 
                  strokeWidth={2}
                />
                <defs>
                  <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
              </AreaChart>
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
            {trendingStocks.slice(0, 5).map((stock, index) => (
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

      {/* AI Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        {/* Market Sentiment */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-success-100 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-success-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">Market Sentiment</h3>
          </div>
          <div className="text-3xl font-bold text-success-600 mb-2">Bullish</div>
          <p className="text-gray-600 text-sm">
            AI analysis indicates positive market sentiment with strong fundamentals.
          </p>
        </div>

        {/* Risk Assessment */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-warning-100 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-warning-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">Risk Level</h3>
          </div>
          <div className="text-3xl font-bold text-warning-600 mb-2">Medium</div>
          <p className="text-gray-600 text-sm">
            Moderate volatility expected. Consider diversification strategies.
          </p>
        </div>

        {/* AI Predictions */}
        <div className="card">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-primary-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900">AI Forecast</h3>
          </div>
          <div className="text-3xl font-bold text-primary-600 mb-2">+2.3%</div>
          <p className="text-gray-600 text-sm">
            Next 30 days outlook based on machine learning models.
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;
