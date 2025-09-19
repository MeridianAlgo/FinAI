import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import EnhancedAnalysis from './pages/EnhancedAnalysis';
import Predictions from './pages/Predictions';
import Portfolio from './pages/Portfolio';
import Market from './pages/Market';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <motion.main
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="pt-16"
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/ai-analysis" element={<EnhancedAnalysis />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/market" element={<Market />} />
          </Routes>
        </motion.main>
      </div>
    </Router>
  );
}

export default App;
