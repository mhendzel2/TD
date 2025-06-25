import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { 
  BarChart3, 
  FileText, 
  Brain, 
  Briefcase, 
  TrendingUp, 
  TrendingDown,
  DollarSign,
  AlertTriangle,
  ChevronRight,
  ChevronDown
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { apiService } from '../../services/apiService';

const Sidebar = ({ collapsed, user }) => {
  const [portfolioSummary, setPortfolioSummary] = useState(null);
  const [watchlist, setWatchlist] = useState([]);
  const [portfolioExpanded, setPortfolioExpanded] = useState(true);
  const [watchlistExpanded, setWatchlistExpanded] = useState(true);

  useEffect(() => {
    // Fetch portfolio summary
    const fetchPortfolioSummary = async () => {
      try {
        const response = await apiService.getPortfolio();
        setPortfolioSummary(response.summary);
      } catch (error) {
        console.error('Failed to fetch portfolio summary:', error);
      }
    };

    // Fetch watchlist (mock data for now)
    const fetchWatchlist = () => {
      setWatchlist([
        { ticker: 'AAPL', price: 175.43, change: 2.34, changePercent: 1.35 },
        { ticker: 'TSLA', price: 248.50, change: -5.20, changePercent: -2.05 },
        { ticker: 'NVDA', price: 875.28, change: 12.45, changePercent: 1.44 },
        { ticker: 'SPY', price: 445.67, change: 1.23, changePercent: 0.28 }
      ]);
    };

    fetchPortfolioSummary();
    fetchWatchlist();

    // Listen for portfolio updates
    const handlePortfolioUpdate = (event) => {
      setPortfolioSummary(event.detail.summary);
    };

    window.addEventListener('portfolioUpdate', handlePortfolioUpdate);

    return () => {
      window.removeEventListener('portfolioUpdate', handlePortfolioUpdate);
    };
  }, []);

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: BarChart3,
      description: 'Overview & Analytics'
    },
    {
      name: 'Newsletters',
      href: '/newsletters',
      icon: FileText,
      description: 'Analysis & Sentiment'
    },
    {
      name: 'Predictions',
      href: '/predictions',
      icon: Brain,
      description: 'ML Probability Scores'
    },
    {
      name: 'Portfolio',
      href: '/portfolio',
      icon: Briefcase,
      description: 'Positions & Risk'
    }
  ];

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
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <aside className={`fixed left-0 top-16 h-[calc(100vh-4rem)] bg-gray-800 border-r border-gray-700 transition-all duration-300 z-30 ${
      collapsed ? 'w-16' : 'w-72'
    }`}>
      <div className="flex flex-col h-full">
        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:text-white hover:bg-gray-700'
                  }`
                }
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!collapsed && (
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{item.name}</p>
                    <p className="text-xs text-gray-400 truncate">{item.description}</p>
                  </div>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Portfolio Summary */}
        {!collapsed && portfolioSummary && (
          <div className="p-4 border-t border-gray-700">
            <Card className="bg-gray-750 border-gray-600">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-gray-300">
                    Portfolio Summary
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setPortfolioExpanded(!portfolioExpanded)}
                    className="h-6 w-6 p-0 text-gray-400 hover:text-white"
                  >
                    {portfolioExpanded ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </CardHeader>
              
              {portfolioExpanded && (
                <CardContent className="pt-0 space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Total Value</span>
                    <span className="text-sm font-medium text-white">
                      {formatCurrency(portfolioSummary.total_value)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">P&L</span>
                    <div className="flex items-center space-x-1">
                      {portfolioSummary.total_pnl >= 0 ? (
                        <TrendingUp className="h-3 w-3 text-green-400" />
                      ) : (
                        <TrendingDown className="h-3 w-3 text-red-400" />
                      )}
                      <span className={`text-sm font-medium ${
                        portfolioSummary.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {formatCurrency(portfolioSummary.total_pnl)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Positions</span>
                    <Badge variant="secondary" className="text-xs">
                      {portfolioSummary.position_count}
                    </Badge>
                  </div>

                  {portfolioSummary.risk_metrics?.concentration_risk > 0.3 && (
                    <div className="flex items-center space-x-1 text-yellow-400">
                      <AlertTriangle className="h-3 w-3" />
                      <span className="text-xs">High concentration risk</span>
                    </div>
                  )}
                </CardContent>
              )}
            </Card>
          </div>
        )}

        {/* Watchlist */}
        {!collapsed && (
          <div className="p-4 border-t border-gray-700">
            <Card className="bg-gray-750 border-gray-600">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-gray-300">
                    Watchlist
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setWatchlistExpanded(!watchlistExpanded)}
                    className="h-6 w-6 p-0 text-gray-400 hover:text-white"
                  >
                    {watchlistExpanded ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </CardHeader>
              
              {watchlistExpanded && (
                <CardContent className="pt-0 space-y-2">
                  {watchlist.map((item) => (
                    <div key={item.ticker} className="flex justify-between items-center">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-white">{item.ticker}</p>
                        <p className="text-xs text-gray-400">${item.price}</p>
                      </div>
                      <div className="text-right">
                        <div className={`flex items-center space-x-1 ${
                          item.change >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {item.change >= 0 ? (
                            <TrendingUp className="h-3 w-3" />
                          ) : (
                            <TrendingDown className="h-3 w-3" />
                          )}
                          <span className="text-xs">
                            {formatPercent(item.changePercent)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              )}
            </Card>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;

