import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Brain, 
  FileText, 
  Briefcase, 
  AlertTriangle,
  DollarSign,
  Target,
  Activity,
  RefreshCw
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { apiService } from '../../services/apiService';
import LoadingSpinner from '../ui/LoadingSpinner';

const MainDashboard = ({ user }) => {
  const { isConnected, subscribe } = useWebSocket();
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [refreshing, setRefreshing] = useState(false);

  // Mock performance data for charts
  const [performanceData] = useState([
    { date: '2024-01-01', portfolio: 100000, benchmark: 100000 },
    { date: '2024-02-01', portfolio: 102500, benchmark: 101200 },
    { date: '2024-03-01', portfolio: 105800, benchmark: 103500 },
    { date: '2024-04-01', portfolio: 108200, benchmark: 105100 },
    { date: '2024-05-01', portfolio: 112400, benchmark: 106800 },
    { date: '2024-06-01', portfolio: 115600, benchmark: 108200 }
  ]);

  useEffect(() => {
    fetchDashboardData();

    // Subscribe to real-time updates
    const unsubscribePrediction = subscribe('predictionUpdate', handlePredictionUpdate);
    const unsubscribePortfolio = subscribe('portfolioUpdate', handlePortfolioUpdate);
    const unsubscribeAlert = subscribe('newAlert', handleNewAlert);

    return () => {
      unsubscribePrediction();
      unsubscribePortfolio();
      unsubscribeAlert();
    };
  }, []);

  const fetchDashboardData = async () => {
    try {
      setRefreshing(true);
      const [statsResponse, portfolioResponse, predictionsResponse] = await Promise.all([
        apiService.getDashboardStats(),
        apiService.getPortfolio(),
        apiService.getTopPredictions({ limit: 5, min_probability: 0.7 })
      ]);

      setDashboardData({
        stats: statsResponse.stats,
        recentPredictions: statsResponse.recent_high_probability_predictions,
        portfolio: portfolioResponse.summary,
        topPredictions: predictionsResponse.top_predictions
      });

      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handlePredictionUpdate = (data) => {
    // Update predictions in real-time
    setDashboardData(prev => ({
      ...prev,
      recentPredictions: [data, ...(prev?.recentPredictions || [])].slice(0, 5)
    }));
  };

  const handlePortfolioUpdate = (data) => {
    // Update portfolio data in real-time
    setDashboardData(prev => ({
      ...prev,
      portfolio: data.summary
    }));
  };

  const handleNewAlert = (data) => {
    // Handle new alerts (could show toast notification)
    console.log('New alert received:', data);
  };

  const formatCurrency = (value) => {
    if (!value) return '$0.00';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value) => {
    if (!value) return '0.00%';
    return `${(value * 100).toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">
            Welcome back, {user?.first_name || 'Trader'}
          </h1>
          <p className="text-gray-400 mt-1">
            Here's what's happening with your trading dashboard today.
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm text-gray-400">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
            <span>{isConnected ? 'Live Data' : 'Offline'}</span>
          </div>
          
          <Button
            variant="outline"
            size="sm"
            onClick={fetchDashboardData}
            disabled={refreshing}
            className="border-gray-600 text-gray-300 hover:text-white hover:bg-gray-700"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">
              Portfolio Value
            </CardTitle>
            <DollarSign className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {formatCurrency(dashboardData?.portfolio?.total_value)}
            </div>
            <p className="text-xs text-gray-400">
              {dashboardData?.portfolio?.total_pnl >= 0 ? '+' : ''}
              {formatCurrency(dashboardData?.portfolio?.total_pnl)} today
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">
              Active Predictions
            </CardTitle>
            <Brain className="h-4 w-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {dashboardData?.stats?.predictions_count || 0}
            </div>
            <p className="text-xs text-gray-400">
              {dashboardData?.recentPredictions?.length || 0} high probability
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">
              Newsletters Analyzed
            </CardTitle>
            <FileText className="h-4 w-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {dashboardData?.stats?.newsletters_count || 0}
            </div>
            <p className="text-xs text-gray-400">
              This month
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">
              Open Positions
            </CardTitle>
            <Briefcase className="h-4 w-4 text-orange-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {dashboardData?.stats?.positions_count || 0}
            </div>
            <p className="text-xs text-gray-400">
              {dashboardData?.stats?.unread_alerts || 0} alerts pending
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts and Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Performance Chart */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Portfolio Performance</CardTitle>
            <CardDescription className="text-gray-400">
              Your portfolio vs benchmark over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="date" 
                  stroke="#9CA3AF"
                  fontSize={12}
                  tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short' })}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  fontSize={12}
                  tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#F3F4F6'
                  }}
                  formatter={(value) => [formatCurrency(value), '']}
                  labelFormatter={(label) => new Date(label).toLocaleDateString()}
                />
                <Line 
                  type="monotone" 
                  dataKey="portfolio" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  name="Portfolio"
                />
                <Line 
                  type="monotone" 
                  dataKey="benchmark" 
                  stroke="#6B7280" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Benchmark"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Top Predictions */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Top ML Predictions</CardTitle>
            <CardDescription className="text-gray-400">
              Highest probability trading opportunities
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={dashboardData?.topPredictions || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="ticker" 
                  stroke="#9CA3AF"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  fontSize={12}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#F3F4F6'
                  }}
                  formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Probability']}
                />
                <Bar 
                  dataKey="probability_score" 
                  fill="#10B981"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent High-Probability Predictions */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Recent High-Probability Predictions</CardTitle>
            <CardDescription className="text-gray-400">
              Latest ML predictions with high confidence
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dashboardData?.recentPredictions?.length > 0 ? (
                dashboardData.recentPredictions.map((prediction, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-750 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                        <span className="text-white font-bold text-sm">
                          {prediction.ticker}
                        </span>
                      </div>
                      <div>
                        <p className="text-white font-medium">{prediction.ticker}</p>
                        <p className="text-xs text-gray-400">
                          {new Date(prediction.prediction_timestamp).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge 
                        variant={prediction.probability_score >= 0.8 ? "default" : "secondary"}
                        className={prediction.probability_score >= 0.8 ? "bg-green-600" : "bg-yellow-600"}
                      >
                        {formatPercent(prediction.probability_score)}
                      </Badge>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <Target className="h-12 w-12 text-gray-600 mx-auto mb-4" />
                  <p className="text-gray-400">No recent high-probability predictions</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* System Status */}
        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">System Status</CardTitle>
            <CardDescription className="text-gray-400">
              Real-time system health and connectivity
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-green-400" />
                  <span className="text-gray-300">WebSocket Connection</span>
                </div>
                <Badge variant={isConnected ? "default" : "destructive"}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </Badge>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Brain className="h-4 w-4 text-blue-400" />
                  <span className="text-gray-300">ML Engine</span>
                </div>
                <Badge variant="default">
                  Operational
                </Badge>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileText className="h-4 w-4 text-purple-400" />
                  <span className="text-gray-300">Newsletter Processing</span>
                </div>
                <Badge variant="default">
                  Active
                </Badge>
              </div>

              <div className="pt-4 border-t border-gray-700">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Last Update</span>
                  <span className="text-gray-300">
                    {lastUpdate.toLocaleTimeString()}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default MainDashboard;
