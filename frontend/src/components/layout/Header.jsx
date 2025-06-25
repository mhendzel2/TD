import React, { useState, useEffect } from 'react';
import { Bell, Menu, Search, Settings, LogOut, User, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useAuth } from '../../contexts/AuthContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { apiService } from '../../services/apiService';

const Header = ({ user, onToggleSidebar }) => {
  const { logout } = useAuth();
  const { isConnected } = useWebSocket();
  const [unreadAlerts, setUnreadAlerts] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    // Fetch unread alerts count
    const fetchUnreadAlerts = async () => {
      try {
        const response = await apiService.getUserAlerts({ unread_only: true });
        setUnreadAlerts(response.unread_count || 0);
      } catch (error) {
        console.error('Failed to fetch unread alerts:', error);
      }
    };

    fetchUnreadAlerts();

    // Listen for new alerts
    const handleNewAlert = () => {
      setUnreadAlerts(prev => prev + 1);
    };

    window.addEventListener('newAlert', handleNewAlert);

    return () => {
      window.removeEventListener('newAlert', handleNewAlert);
    };
  }, []);

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      // Implement search functionality
      console.log('Searching for:', searchQuery);
    }
  };

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left section */}
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSidebar}
            className="text-gray-300 hover:text-white hover:bg-gray-700"
          >
            <Menu className="h-5 w-5" />
          </Button>

          <div className="flex items-center space-x-2">
            <TrendingUp className="h-6 w-6 text-blue-400" />
            <h1 className="text-xl font-bold text-white">Trading Dashboard</h1>
          </div>

          {/* Connection status indicator */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              isConnected ? 'bg-green-400' : 'bg-red-400'
            }`} />
            <span className="text-xs text-gray-400">
              {isConnected ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Center section - Search */}
        <div className="flex-1 max-w-md mx-8">
          <form onSubmit={handleSearch} className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              type="text"
              placeholder="Search tickers, newsletters..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:border-blue-400"
            />
          </form>
        </div>

        {/* Right section */}
        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="relative text-gray-300 hover:text-white hover:bg-gray-700">
                <Bell className="h-5 w-5" />
                {unreadAlerts > 0 && (
                  <Badge 
                    variant="destructive" 
                    className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center text-xs p-0"
                  >
                    {unreadAlerts > 99 ? '99+' : unreadAlerts}
                  </Badge>
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-80 bg-gray-800 border-gray-700">
              <DropdownMenuLabel className="text-white">Notifications</DropdownMenuLabel>
              <DropdownMenuSeparator className="bg-gray-700" />
              {unreadAlerts > 0 ? (
                <div className="p-2">
                  <p className="text-sm text-gray-300">
                    You have {unreadAlerts} unread alert{unreadAlerts !== 1 ? 's' : ''}
                  </p>
                  <Button 
                    variant="link" 
                    size="sm" 
                    className="text-blue-400 p-0 h-auto"
                    onClick={() => {
                      // Navigate to alerts page
                      window.location.href = '/alerts';
                    }}
                  >
                    View all alerts
                  </Button>
                </div>
              ) : (
                <div className="p-4 text-center">
                  <p className="text-sm text-gray-400">No new notifications</p>
                </div>
              )}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* User menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white hover:bg-gray-700">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <User className="h-4 w-4 text-white" />
                  </div>
                  <span className="hidden md:block">
                    {user?.first_name || user?.email?.split('@')[0] || 'User'}
                  </span>
                </div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 bg-gray-800 border-gray-700">
              <DropdownMenuLabel className="text-white">
                <div>
                  <p className="font-medium">{user?.first_name} {user?.last_name}</p>
                  <p className="text-sm text-gray-400">{user?.email}</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator className="bg-gray-700" />
              
              <DropdownMenuItem className="text-gray-300 hover:text-white hover:bg-gray-700">
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              
              <DropdownMenuItem className="text-gray-300 hover:text-white hover:bg-gray-700">
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </DropdownMenuItem>
              
              <DropdownMenuSeparator className="bg-gray-700" />
              
              <DropdownMenuItem 
                onClick={handleLogout}
                className="text-red-400 hover:text-red-300 hover:bg-gray-700"
              >
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
};

export default Header;

