import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Plus, 
  TrendingUp, 
  TrendingDown, 
  PieChart,
  BarChart3,
  Target,
  Shield,
  DollarSign,
  Activity,
  AlertTriangle,
  CheckCircle,
  Trash2,
  Edit3
} from 'lucide-react';
import { PieChart as RechartsPieChart, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Pie } from 'recharts';
import { toast } from 'react-hot-toast';
import api from '../services/api';

const Portfolio = () => {
  const [portfolio, setPortfolio] = useState([]);
  const [optimization, setOptimization] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [newWeight, setNewWeight] = useState('');

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'];

  useEffect(() => {
    // Load saved portfolio from localStorage
    const savedPortfolio = localStorage.getItem('portfolio');
    if (savedPortfolio) {
      setPortfolio(JSON.parse(savedPortfolio));
    }
  }, []);

  useEffect(() => {
    if (portfolio.length > 0) {
      handleOptimization();
      handlePredictions();
    }
  }, [portfolio]);

  const savePortfolio = (newPortfolio) => {
    setPortfolio(newPortfolio);
    localStorage.setItem('portfolio', JSON.stringify(newPortfolio));
  };

  const addStock = () => {
    if (!newSymbol.trim() || !newWeight.trim()) {
      toast.error('Please enter both symbol and weight');
      return;
    }

    const weight = parseFloat(newWeight);
    if (isNaN(weight) || weight <= 0 || weight > 100) {
      toast.error('Weight must be a number between 0 and 100');
      return;
    }

    const totalWeight = portfolio.reduce((sum, item) => sum + item.weight, 0);
    if (totalWeight + weight > 100) {
      toast.error('Total portfolio weight cannot exceed 100%');
      return;
    }

    const newStock = {
      symbol: newSymbol.toUpperCase(),
      weight: weight,
      id: Date.now()
    };

    savePortfolio([...portfolio, newStock]);
    setNewSymbol('');
    setNewWeight('');
    setShowAddModal(false);
    toast.success('Stock added to portfolio');
  };

  const removeStock = (id) => {
    const newPortfolio = portfolio.filter(stock => stock.id !== id);
    savePortfolio(newPortfolio);
    toast.success('Stock removed from portfolio');
  };

  const updateWeight = (id, newWeight) => {
    const weight = parseFloat(newWeight);
    if (isNaN(weight) || weight <= 0 || weight > 100) {
      toast.error('Weight must be a number between 0 and 100');
      return;
    }

    const totalWeight = portfolio.reduce((sum, item) => sum + (item.id === id ? weight : item.weight), 0);
    if (totalWeight > 100) {
      toast.error('Total portfolio weight cannot exceed 100%');
      return;
    }

    const newPortfolio = portfolio.map(stock => 
      stock.id === id ? { ...stock, weight: weight } : stock
    );
    savePortfolio(newPortfolio);
  };

  const handleOptimization = async () => {
    if (portfolio.length < 2) return;

    try {
      setLoading(true);
      const symbols = portfolio.map(stock => stock.symbol);
      const weights = portfolio.map(stock => stock.weight / 100); // Convert to decimal
      
      const response = await api.get('/api/analysis/portfolio/optimization', {
        params: {
          symbols: symbols.join(','),
          risk_tolerance: 'medium'
        }
      });
      
      setOptimization(response.data);
    } catch (error) {
      console.error('Optimization error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePredictions = async () => {
    if (portfolio.length === 0) return;

    try {
      const symbols = portfolio.map(stock => stock.symbol);
      const weights = portfolio.map(stock => stock.weight / 100);
      
      const response = await api.post('/api/predictions/portfolio/returns', {
        symbols: symbols,
        weights: weights,
        prediction_days: 30
      });
      
      setPredictions(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  const preparePieData = () => {
    return portfolio.map((stock, index) => ({
      name: stock.symbol,
      value: stock.weight,
      color: COLORS[index % COLORS.length]
    }));
  };

  const preparePerformanceData = () => {
    if (!predictions) return [];
    
    return predictions.predicted_returns.predicted_daily_returns.map((return_, index) => ({
      day: index + 1,
      return: return_ * 100,
      cumulative: predictions.predicted_returns.predicted_daily_returns
        .slice(0, index + 1)
        .reduce((sum, r) => sum + r, 0) * 100
    }));
  };

  const getTotalWeight = () => {
    return portfolio.reduce((sum, stock) => sum + stock.weight, 0);
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
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Portfolio Management</h1>
            <p className="text-gray-600">AI-powered portfolio optimization and analysis</p>
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn btn-primary flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Add Stock</span>
          </button>
        </div>
      </motion.div>

      {/* Portfolio Overview */}
      {portfolio.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8"
        >
          {/* Portfolio Allocation */}
          <div className="lg:col-span-2 card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Portfolio Allocation</h2>
            
            {portfolio.length > 0 && (
              <div className="h-64 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsPieChart>
                    <Pie
                      data={preparePieData()}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {preparePieData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </div>
            )}
            
            <div className="space-y-3">
              {portfolio.map((stock, index) => (
                <div key={stock.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div 
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    ></div>
                    <span className="font-medium text-gray-900">{stock.symbol}</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <input
                      type="number"
                      value={stock.weight}
                      onChange={(e) => updateWeight(stock.id, e.target.value)}
                      className="w-20 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
                      min="0"
                      max="100"
                      step="0.1"
                    />
                    <span className="text-sm text-gray-600">%</span>
                    <button
                      onClick={() => removeStock(stock.id)}
                      className="p-1 text-gray-400 hover:text-danger-600 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-4 p-4 bg-primary-50 rounded-lg">
              <div className="flex justify-between items-center">
                <span className="font-medium text-primary-900">Total Allocation</span>
                <span className="text-xl font-bold text-primary-900">
                  {getTotalWeight().toFixed(1)}%
                </span>
              </div>
              {getTotalWeight() < 100 && (
                <p className="text-sm text-primary-700 mt-1">
                  {100 - getTotalWeight().toFixed(1)}% available for allocation
                </p>
              )}
            </div>
          </div>

          {/* Portfolio Metrics */}
          <div className="space-y-6">
            {optimization && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Optimization Results</h3>
                <div className="space-y-4">
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Expected Return</h4>
                    <div className="text-2xl font-bold text-success-600">
                      {(optimization.expected_return * 100).toFixed(2)}%
                    </div>
                  </div>
                  
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Expected Volatility</h4>
                    <div className="text-2xl font-bold text-warning-600">
                      {(optimization.expected_volatility * 100).toFixed(2)}%
                    </div>
                  </div>
                  
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Sharpe Ratio</h4>
                    <div className="text-2xl font-bold text-primary-600">
                      {optimization.sharpe_ratio?.toFixed(2) || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {predictions && (
              <div className="card">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">30-Day Forecast</h3>
                <div className="space-y-4">
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Expected Return</h4>
                    <div className="text-2xl font-bold text-success-600">
                      {(predictions.predicted_returns.expected_total_return * 100).toFixed(2)}%
                    </div>
                  </div>
                  
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Annual Return</h4>
                    <div className="text-2xl font-bold text-primary-600">
                      {(predictions.predicted_returns.expected_annual_return * 100).toFixed(2)}%
                    </div>
                  </div>
                  
                  <div className="metric-card">
                    <h4 className="font-medium text-gray-900 mb-2">Volatility</h4>
                    <div className="text-2xl font-bold text-warning-600">
                      {(predictions.predicted_returns.expected_volatility * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Performance Charts */}
      {predictions && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8"
        >
          {/* Returns Chart */}
          <div className="card">
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Predicted Returns</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={preparePerformanceData()}>
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
                    formatter={(value, name) => [
                      `${value.toFixed(2)}%`,
                      name === 'return' ? 'Daily Return' : 'Cumulative Return'
                    ]}
                  />
                  <Line
                    type="monotone"
                    dataKey="return"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="cumulative"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Risk Metrics */}
          <div className="card">
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Risk Analysis</h3>
            {predictions.risk_metrics && (
              <div className="space-y-4">
                <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Value at Risk (95%)</span>
                  <span className="font-medium text-danger-600">
                    {(predictions.risk_metrics.var_95 * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Value at Risk (99%)</span>
                  <span className="font-medium text-danger-600">
                    {(predictions.risk_metrics.var_99 * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Maximum Drawdown</span>
                  <span className="font-medium text-warning-600">
                    {(predictions.risk_metrics.max_drawdown * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                  <span className="text-sm text-gray-600">Risk Score</span>
                  <span className={`font-medium ${getRiskColor(
                    predictions.risk_metrics.risk_score > 30 ? 'high' :
                    predictions.risk_metrics.risk_score > 15 ? 'medium' : 'low'
                  )}`}>
                    {predictions.risk_metrics.risk_score.toFixed(1)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Optimization Suggestions */}
      {optimization && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Optimization Suggestions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Optimal Weights</h4>
              <div className="space-y-2">
                {Object.entries(optimization.optimal_weights).map(([symbol, weight], index) => (
                  <div key={symbol} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="font-medium">{symbol}</span>
                    <span className="text-primary-600 font-semibold">
                      {(weight * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Recommendations</h4>
              <div className="space-y-3">
                {predictions?.optimization_suggestions?.map((suggestion, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                    <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5" />
                    <span className="text-sm text-blue-800">{suggestion}</span>
                  </div>
                )) || (
                  <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                    <span className="text-sm text-green-800">
                      Your portfolio is well-diversified and optimized for your risk tolerance.
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Empty State */}
      {portfolio.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-12"
        >
          <PieChart className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No stocks in portfolio</h3>
          <p className="text-gray-600 mb-6">Add stocks to your portfolio to start analyzing and optimizing</p>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn btn-primary flex items-center space-x-2 mx-auto"
          >
            <Plus className="w-4 h-4" />
            <span>Add Your First Stock</span>
          </button>
        </motion.div>
      )}

      {/* Add Stock Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-xl p-6 w-full max-w-md mx-4"
          >
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Add Stock to Portfolio</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Stock Symbol
                </label>
                <input
                  type="text"
                  placeholder="e.g., AAPL, MSFT, GOOGL"
                  value={newSymbol}
                  onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                  className="input"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Weight (%)
                </label>
                <input
                  type="number"
                  placeholder="e.g., 25.5"
                  value={newWeight}
                  onChange={(e) => setNewWeight(e.target.value)}
                  className="input"
                  min="0"
                  max="100"
                  step="0.1"
                />
              </div>
            </div>
            
            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => setShowAddModal(false)}
                className="btn btn-secondary flex-1"
              >
                Cancel
              </button>
              <button
                onClick={addStock}
                className="btn btn-primary flex-1"
              >
                Add Stock
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default Portfolio;
