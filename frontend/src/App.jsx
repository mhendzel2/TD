import React, { useState, useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';

import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeContext, useTheme } from './contexts/ThemeContext';
import { WebSocketProvider } from './contexts/WebSocketContext';

import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import LoadingSpinner from './components/ui/LoadingSpinner';

// Lazy load pages for better performance
const LoginPage = lazy(() => import('./pages/LoginPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const NewsletterPage = lazy(() => import('./pages/NewsletterPage'));
const MLAnalysisPage = lazy(() => import('./pages/MLAnalysisPage'));
const PortfolioPage = lazy(() => import('./pages/PortfolioPage'));
const AlertsPage = lazy(() => import('./pages/AlertsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));

function PrivateRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return isAuthenticated ? children : <Navigate to="/login" />;
}

function AppContent() {
  const { theme } = useTheme();
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('info');

  // Global error handler (can be improved with a dedicated error context)
  useEffect(() => {
    const handleError = (event) => {
      console.error('Unhandled error:', event.error);
      setSnackbarMessage(event.message || 'An unexpected error occurred.');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    };

    window.addEventListener('error', handleError);
    return () => {
      window.removeEventListener('error', handleError);
    };
  }, []);

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex', height: '100vh' }}>
        <Header />
        <Sidebar />
        <Box component="main" sx={{ flexGrow: 1, p: 3, mt: '64px', overflowY: 'auto' }}>
          <Suspense fallback={<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}><CircularProgress /></Box>}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/dashboard" element={<PrivateRoute><DashboardPage /></PrivateRoute>} />
              <Route path="/newsletter" element={<PrivateRoute><NewsletterPage /></PrivateRoute>} />
              <Route path="/ml-analysis" element={<PrivateRoute><MLAnalysisPage /></PrivateRoute>} />
              <Route path="/portfolio" element={<PrivateRoute><PortfolioPage /></PrivateRoute>} />
              <Route path="/alerts" element={<PrivateRoute><AlertsPage /></PrivateRoute>} />
              <Route path="/settings" element={<PrivateRoute><SettingsPage /></PrivateRoute>} />
              <Route path="/" element={<Navigate to="/dashboard" />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </Box>
      </Box>
      <Snackbar open={snackbarOpen} autoHideDuration={6000} onClose={handleSnackbarClose}>
        <Alert onClose={handleSnackbarClose} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <ThemeContext>
          <WebSocketProvider>
            <AppContent />
          </WebSocketProvider>
        </ThemeContext>
      </AuthProvider>
    </Router>
  );
}

export default App;


