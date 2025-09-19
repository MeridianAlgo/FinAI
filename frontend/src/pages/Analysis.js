import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  TrendingUp, 
  TrendingDown, 
  Activity,
  BarChart3,
  PieChart,
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  Shield,
  Brain
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart as RechartsPieChart, Cell } from 'recharts';
import { toast } from 'react-hot-toast';
import api from '../services/api';

const Analysis = () => {
  const [searchSymbol, setSearchSymbol] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState('comprehensive');

  const analysisTypes = [
    { id: 'comprehensive', name: 'Comprehensive', icon: BarChart3 },
    { id: 'technical', name: 'Technical', icon: TrendingUp },
    { id: 'fundamental', name: 'Fundamental', icon: PieChart },
    { id: 'sentiment', name: 'Sentiment', icon: Brain },
    { id: 'risk', name: 'Risk', icon: Shield }
  ];

  const handleAnalysis = async () => {
    if (!searchSymbol.trim()) {
      toast.error('Please enter a stock symbol');
      return;
    }

    try {
      setLoading(true);
      const response = await api.post('/api/analysis/comprehensive', {
        symbol: searchSymbol.toUpperCase(),
        timeframe: '1y',
        analysis_type: selectedAnalysis
      });
      
      setAnalysisData(response.data);
      toast.success('Analysis completed successfully');
    } catch (error) {
      toast.error(error.message || 'Failed to perform analysis');
      console.error('Analysis error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-success-600';
    if (score >= 60) return 'text-success-500';
    if (score >= 40) return 'text-warning-500';
    if (score >= 20) return 'text-danger-500';
    return 'text-danger-600';
  };

  const getScoreBgColor = (score) => {
    if (score >= 80) return 'bg-success-100';
    if (score >= 60) return 'bg-success-50';
    if (score >= 40) return 'bg-warning-50';
    if (score >= 20) return 'bg-danger-50';
    return 'bg-danger-100';
  };

  const getRecommendationColor = (recommendation) => {
    switch (recommendation) {
      case 'Strong Buy':
        return 'status-success';
      case 'Buy':
        return 'status-success';
      case 'Hold':
        return 'status-info';
      case 'Sell':
        return 'status-warning';
      case 'Strong Sell':
        return 'status-danger';
      default:
        return 'status-info';
    }
  };

  const formatNumber = (num) => {
    if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(2);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Financial Analysis</h1>
        <p className="text-gray-600">AI-powered comprehensive financial analysis and insights</p>
      </motion.div>

      {/* Search Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card mb-8"
      >
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
                value={searchSymbol}
                onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                className="input pl-10"
                onKeyPress={(e) => e.key === 'Enter' && handleAnalysis()}
              />
            </div>
          </div>
          <div className="flex gap-2">
            <select
              value={selectedAnalysis}
              onChange={(e) => setSelectedAnalysis(e.target.value)}
              className="input"
            >
              {analysisTypes.map((type) => (
                <option key={type.id} value={type.id}>
                  {type.name} Analysis
                </option>
              ))}
            </select>
            <button
              onClick={handleAnalysis}
              disabled={loading}
              className="btn btn-primary flex items-center space-x-2"
            >
              {loading ? (
                <div className="spinner w-4 h-4"></div>
              ) : (
                <BarChart3 className="w-4 h-4" />
              )}
              <span>Analyze</span>
            </button>
          </div>
        </div>
      </motion.div>

      {/* Analysis Results */}
      {analysisData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-8"
        >
          {/* Overall Score */}
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Analysis Results for {analysisData.symbol}
              </h2>
              <div className={`status-indicator ${getRecommendationColor(analysisData.recommendation)}`}>
                {analysisData.recommendation}
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className={`w-20 h-20 mx-auto rounded-full ${getScoreBgColor(analysisData.overall_score)} flex items-center justify-center mb-3`}>
                  <span className={`text-2xl font-bold ${getScoreColor(analysisData.overall_score)}`}>
                    {analysisData.overall_score}
                  </span>
                </div>
                <h3 className="font-semibold text-gray-900">Overall Score</h3>
                <p className="text-sm text-gray-600">Out of 100</p>
              </div>
              
              <div className="text-center">
                <div className={`w-20 h-20 mx-auto rounded-full ${getScoreBgColor(analysisData.technical_analysis?.technical_score || 50)} flex items-center justify-center mb-3`}>
                  <span className={`text-2xl font-bold ${getScoreColor(analysisData.technical_analysis?.technical_score || 50)}`}>
                    {analysisData.technical_analysis?.technical_score || 50}
                  </span>
                </div>
                <h3 className="font-semibold text-gray-900">Technical</h3>
                <p className="text-sm text-gray-600">Chart Analysis</p>
              </div>
              
              <div className="text-center">
                <div className={`w-20 h-20 mx-auto rounded-full ${getScoreBgColor(analysisData.fundamental_analysis?.fundamental_score || 50)} flex items-center justify-center mb-3`}>
                  <span className={`text-2xl font-bold ${getScoreColor(analysisData.fundamental_analysis?.fundamental_score || 50)}`}>
                    {analysisData.fundamental_analysis?.fundamental_score || 50}
                  </span>
                </div>
                <h3 className="font-semibold text-gray-900">Fundamental</h3>
                <p className="text-sm text-gray-600">Financial Health</p>
              </div>
              
              <div className="text-center">
                <div className={`w-20 h-20 mx-auto rounded-full ${getScoreBgColor(analysisData.sentiment_analysis?.overall_sentiment_score || 50)} flex items-center justify-center mb-3`}>
                  <span className={`text-2xl font-bold ${getScoreColor(analysisData.sentiment_analysis?.overall_sentiment_score || 50)}`}>
                    {analysisData.sentiment_analysis?.overall_sentiment_score || 50}
                  </span>
                </div>
                <h3 className="font-semibold text-gray-900">Sentiment</h3>
                <p className="text-sm text-gray-600">Market Mood</p>
              </div>
            </div>
          </div>

          {/* Technical Analysis */}
          {analysisData.technical_analysis && (
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <TrendingUp className="w-5 h-5 text-primary-600" />
                <span>Technical Analysis</span>
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Current Price</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    ${analysisData.technical_analysis.current_price?.toFixed(2) || 'N/A'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">7-Day Change</h4>
                  <div className={`text-2xl font-bold flex items-center space-x-2 ${
                    analysisData.technical_analysis.price_change_7d?.change_percent > 0 ? 'text-success-600' : 'text-danger-600'
                  }`}>
                    {analysisData.technical_analysis.price_change_7d?.change_percent > 0 ? (
                      <TrendingUp className="w-5 h-5" />
                    ) : (
                      <TrendingDown className="w-5 h-5" />
                    )}
                    <span>
                      {analysisData.technical_analysis.price_change_7d?.change_percent > 0 ? '+' : ''}
                      {analysisData.technical_analysis.price_change_7d?.change_percent?.toFixed(2) || '0.00'}%
                    </span>
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">30-Day Change</h4>
                  <div className={`text-2xl font-bold flex items-center space-x-2 ${
                    analysisData.technical_analysis.price_change_30d?.change_percent > 0 ? 'text-success-600' : 'text-danger-600'
                  }`}>
                    {analysisData.technical_analysis.price_change_30d?.change_percent > 0 ? (
                      <TrendingUp className="w-5 h-5" />
                    ) : (
                      <TrendingDown className="w-5 h-5" />
                    )}
                    <span>
                      {analysisData.technical_analysis.price_change_30d?.change_percent > 0 ? '+' : ''}
                      {analysisData.technical_analysis.price_change_30d?.change_percent?.toFixed(2) || '0.00'}%
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Technical Indicators */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Moving Averages</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-600">Signal</span>
                      <span className={`status-indicator ${
                        analysisData.technical_analysis.moving_averages?.signal === 'bullish' ? 'status-success' :
                        analysisData.technical_analysis.moving_averages?.signal === 'bearish' ? 'status-danger' : 'status-info'
                      }`}>
                        {analysisData.technical_analysis.moving_averages?.signal || 'neutral'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-600">SMA 20</span>
                      <span className="font-medium">
                        ${analysisData.technical_analysis.moving_averages?.sma_20?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-600">SMA 50</span>
                      <span className="font-medium">
                        ${analysisData.technical_analysis.moving_averages?.sma_50?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">RSI Analysis</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-600">RSI Value</span>
                      <span className="font-medium">
                        {analysisData.technical_analysis.rsi_analysis?.rsi?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <span className="text-sm text-gray-600">Signal</span>
                      <span className={`status-indicator ${
                        analysisData.technical_analysis.rsi_analysis?.signal === 'oversold' ? 'status-success' :
                        analysisData.technical_analysis.rsi_analysis?.signal === 'overbought' ? 'status-danger' : 'status-info'
                      }`}>
                        {analysisData.technical_analysis.rsi_analysis?.signal || 'neutral'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Fundamental Analysis */}
          {analysisData.fundamental_analysis && (
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <PieChart className="w-5 h-5 text-primary-600" />
                <span>Fundamental Analysis</span>
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">P/E Ratio</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {analysisData.fundamental_analysis.valuation_metrics?.pe_ratio?.toFixed(2) || 'N/A'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Market Cap</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {formatNumber(analysisData.fundamental_analysis.valuation_metrics?.market_cap || 0)}
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Profit Margin</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {((analysisData.fundamental_analysis.profitability_metrics?.profit_margin || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">ROE</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {((analysisData.fundamental_analysis.profitability_metrics?.return_on_equity || 0) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Sentiment Analysis */}
          {analysisData.sentiment_analysis && (
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <Brain className="w-5 h-5 text-primary-600" />
                <span>Sentiment Analysis</span>
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">News Sentiment</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {analysisData.sentiment_analysis.news_sentiment?.sentiment || 'neutral'}
                  </div>
                  <div className="text-sm text-gray-600">
                    Score: {analysisData.sentiment_analysis.news_sentiment?.score?.toFixed(1) || '50.0'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Social Sentiment</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {analysisData.sentiment_analysis.social_sentiment?.sentiment || 'neutral'}
                  </div>
                  <div className="text-sm text-gray-600">
                    Score: {analysisData.sentiment_analysis.social_sentiment?.score?.toFixed(1) || '50.0'}
                  </div>
                </div>
                
                <div className="metric-card">
                  <h4 className="font-medium text-gray-900 mb-2">Analyst Consensus</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {analysisData.sentiment_analysis.analyst_sentiment?.consensus || 'Hold'}
                  </div>
                  <div className="text-sm text-gray-600">
                    {analysisData.sentiment_analysis.analyst_sentiment?.recommendation_count || 0} recommendations
                  </div>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default Analysis;
