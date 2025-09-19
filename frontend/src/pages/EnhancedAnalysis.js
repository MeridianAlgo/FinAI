import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  Brain, 
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
  Zap,
  Star,
  Award,
  TrendingUp as UpIcon,
  TrendingDown as DownIcon
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart as RechartsPieChart, Cell } from 'recharts';
import { toast } from 'react-hot-toast';
import api from '../services/api';

const EnhancedAnalysis = () => {
  const [searchSymbol, setSearchSymbol] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await api.get('/api/ai/model-status');
      setModelStatus(response.data);
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const handleAnalysis = async () => {
    if (!searchSymbol.trim()) {
      toast.error('Please enter a stock symbol');
      return;
    }

    try {
      setLoading(true);
      const response = await api.post(`/api/ai/company/${searchSymbol.toUpperCase()}`, {
        symbol: searchSymbol.toUpperCase(),
        analysis_type: 'comprehensive'
      });
      
      setAnalysisData(response.data);
      toast.success('AI analysis completed successfully');
    } catch (error) {
      toast.error(error.message || 'Failed to perform AI analysis');
      console.error('AI analysis error:', error);
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

  const getTrendIcon = (direction) => {
    if (direction.includes('up') || direction === 'bullish') {
      return <TrendingUp className="w-5 h-5 text-success-600" />;
    } else if (direction.includes('down') || direction === 'bearish') {
      return <TrendingDown className="w-5 h-5 text-danger-600" />;
    } else {
      return <Activity className="w-5 h-5 text-gray-600" />;
    }
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

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'very_positive':
        return 'text-success-600';
      case 'positive':
        return 'text-success-500';
      case 'neutral':
        return 'text-gray-600';
      case 'negative':
        return 'text-danger-500';
      case 'very_negative':
        return 'text-danger-600';
      default:
        return 'text-gray-600';
    }
  };

  const preparePredictionData = () => {
    if (!analysisData?.price_prediction) return [];
    
    const predictions = [];
    const currentPrice = analysisData.price_prediction.current_price;
    const expectedReturn = analysisData.price_prediction.expected_return;
    
    for (let day = 1; day <= 30; day++) {
      const predictedPrice = currentPrice * (1 + expectedReturn * (day / 30));
      predictions.push({
        day: day,
        price: predictedPrice,
        confidence: analysisData.price_prediction.confidence * (1 - day / 100)
      });
    }
    
    return predictions;
  };

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

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
            <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center space-x-3">
              <Brain className="w-8 h-8 text-primary-600" />
              <span>Enhanced AI Analysis</span>
            </h1>
            <p className="text-gray-600">Advanced AI-powered company analysis with machine learning models</p>
          </div>
          {modelStatus && (
            <div className="flex items-center space-x-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${modelStatus.status === 'ready' ? 'bg-success-500' : 'bg-warning-500'}`}></div>
              <span className="text-gray-600">
                AI Models: {modelStatus.status === 'ready' ? 'Ready' : 'Training'}
              </span>
            </div>
          )}
        </div>
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
          <button
            onClick={handleAnalysis}
            disabled={loading}
            className="btn btn-primary flex items-center space-x-2"
          >
            {loading ? (
              <div className="spinner w-4 h-4"></div>
            ) : (
              <Brain className="w-4 h-4" />
            )}
            <span>Analyze with AI</span>
          </button>
        </div>
      </motion.div>

      {/* AI Analysis Results */}
      {analysisData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-8"
        >
          {/* AI Score Overview */}
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
                <Award className="w-6 h-6 text-primary-600" />
                <span>AI Analysis Results for {analysisData.symbol}</span>
              </h2>
              <div className={`status-indicator ${getRecommendationColor(analysisData.recommendation)}`}>
                {analysisData.recommendation}
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Overall AI Score */}
              <div className="text-center">
                <div className={`w-32 h-32 mx-auto rounded-full ${getScoreBgColor(analysisData.ai_score)} flex items-center justify-center mb-4`}>
                  <span className={`text-4xl font-bold ${getScoreColor(analysisData.ai_score)}`}>
                    {analysisData.ai_score.toFixed(1)}
                  </span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900">AI Score</h3>
                <p className="text-sm text-gray-600">Out of 100</p>
                <div className="mt-2">
                  <div className="flex items-center justify-center space-x-1">
                    {[...Array(5)].map((_, i) => (
                      <Star 
                        key={i} 
                        className={`w-4 h-4 ${
                          i < Math.floor(analysisData.ai_score / 20) 
                            ? 'text-yellow-400 fill-current' 
                            : 'text-gray-300'
                        }`} 
                      />
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Confidence Level */}
              <div className="text-center">
                <div className="w-32 h-32 mx-auto rounded-full bg-blue-100 flex items-center justify-center mb-4">
                  <span className="text-4xl font-bold text-blue-600">
                    {(analysisData.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900">Confidence</h3>
                <p className="text-sm text-gray-600">AI Model Confidence</p>
              </div>
              
              {/* Analysis Date */}
              <div className="text-center">
                <div className="w-32 h-32 mx-auto rounded-full bg-purple-100 flex items-center justify-center mb-4">
                  <Clock className="w-12 h-12 text-purple-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900">Analysis Date</h3>
                <p className="text-sm text-gray-600">
                  {new Date(analysisData.analysis_date).toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>

          {/* Detailed Analysis Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Price Prediction */}
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <Target className="w-5 h-5 text-primary-600" />
                <span>Price Prediction</span>
              </h3>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Current Price</h4>
                    <div className="text-2xl font-bold text-gray-900">
                      ${analysisData.price_prediction.current_price?.toFixed(2) || 'N/A'}
                    </div>
                  </div>
                  
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Expected Return</h4>
                    <div className={`text-2xl font-bold flex items-center space-x-2 ${
                      analysisData.price_prediction.expected_return > 0 ? 'text-success-600' : 'text-danger-600'
                    }`}>
                      {analysisData.price_prediction.expected_return > 0 ? (
                        <UpIcon className="w-5 h-5" />
                      ) : (
                        <DownIcon className="w-5 h-5" />
                      )}
                      <span>
                        {(analysisData.price_prediction.expected_return * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={preparePredictionData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="day" stroke="#6b7280" />
                      <YAxis stroke="#6b7280" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#fff', 
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                        }}
                        formatter={(value) => [`$${value.toFixed(2)}`, 'Predicted Price']}
                      />
                      <Line
                        type="monotone"
                        dataKey="price"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Trend Analysis */}
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <BarChart3 className="w-5 h-5 text-primary-600" />
                <span>Trend Analysis</span>
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Trend Direction</span>
                  <div className="flex items-center space-x-2">
                    {getTrendIcon(analysisData.trend_analysis.direction)}
                    <span className="font-medium capitalize">
                      {analysisData.trend_analysis.direction.replace('_', ' ')}
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Trend Strength</span>
                  <span className="font-medium">
                    {(analysisData.trend_analysis.strength * 100).toFixed(1)}%
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Probability</span>
                  <span className="font-medium">
                    {(analysisData.trend_analysis.probability * 100).toFixed(1)}%
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Confidence</span>
                  <span className="font-medium">
                    {(analysisData.trend_analysis.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Risk Assessment */}
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <Shield className="w-5 h-5 text-primary-600" />
                <span>Risk Assessment</span>
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Risk Level</span>
                  <span className={`font-medium capitalize ${getRiskColor(analysisData.risk_assessment.risk_level)}`}>
                    {analysisData.risk_assessment.risk_level.replace('_', ' ')}
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Risk Score</span>
                  <span className="font-medium">
                    {analysisData.risk_assessment.risk_score.toFixed(1)}/100
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Predicted Volatility</span>
                  <span className="font-medium">
                    {(analysisData.risk_assessment.predicted_volatility * 100).toFixed(2)}%
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Confidence</span>
                  <span className="font-medium">
                    {(analysisData.risk_assessment.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Sentiment Analysis */}
            <div className="card">
              <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
                <PieChart className="w-5 h-5 text-primary-600" />
                <span>Sentiment Analysis</span>
              </h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Sentiment</span>
                  <span className={`font-medium capitalize ${getSentimentColor(analysisData.sentiment_analysis.sentiment)}`}>
                    {analysisData.sentiment_analysis.sentiment.replace('_', ' ')}
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Sentiment Score</span>
                  <span className="font-medium">
                    {analysisData.sentiment_analysis.score.toFixed(1)}/100
                  </span>
                </div>
                
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Confidence</span>
                  <span className="font-medium">
                    {(analysisData.sentiment_analysis.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Fundamental Score */}
          <div className="card">
            <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
              <Award className="w-5 h-5 text-primary-600" />
              <span>Fundamental Analysis</span>
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className={`w-20 h-20 mx-auto rounded-full ${getScoreBgColor(analysisData.fundamental_score.score)} flex items-center justify-center mb-3`}>
                  <span className={`text-2xl font-bold ${getScoreColor(analysisData.fundamental_score.score)}`}>
                    {analysisData.fundamental_score.score.toFixed(1)}
                  </span>
                </div>
                <h4 className="font-medium text-gray-900">Fundamental Score</h4>
                <p className="text-sm text-gray-600">Out of 100</p>
              </div>
              
              <div className="text-center">
                <div className="w-20 h-20 mx-auto rounded-full bg-blue-100 flex items-center justify-center mb-3">
                  <span className="text-2xl font-bold text-blue-600">
                    {analysisData.fundamental_score.grade}
                  </span>
                </div>
                <h4 className="font-medium text-gray-900">Grade</h4>
                <p className="text-sm text-gray-600">AI Assessment</p>
              </div>
              
              <div className="text-center">
                <div className="w-20 h-20 mx-auto rounded-full bg-green-100 flex items-center justify-center mb-3">
                  <span className="text-2xl font-bold text-green-600">
                    {(analysisData.fundamental_score.expected_return * 100).toFixed(1)}%
                  </span>
                </div>
                <h4 className="font-medium text-gray-900">Expected Return</h4>
                <p className="text-sm text-gray-600">Based on Fundamentals</p>
              </div>
            </div>
          </div>

          {/* Key Insights */}
          <div className="card">
            <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center space-x-2">
              <Zap className="w-5 h-5 text-primary-600" />
              <span>AI Key Insights</span>
            </h3>
            
            <div className="space-y-3">
              {analysisData.key_insights.map((insight, index) => (
                <div key={index} className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
                  <CheckCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                  <span className="text-sm text-blue-800">{insight}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default EnhancedAnalysis;
