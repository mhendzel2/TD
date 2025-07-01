import React, { useState, useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate, Outlet } from 'react-router-dom';

import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider as CustomThemeProvider } from './contexts/ThemeContext';
import { WebSocketProvider } from './contexts/WebSocketContext';

import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import LoadingSpinner from './components/ui/LoadingSpinner';
// To handle notifications consistently with shadcn/ui, consider using react-hot-toast
// import { Toaster } from 'react-hot-toast';

// Lazy load pages for better performance
const LoginPage = lazy(() => import('./pages/LoginPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const NewsletterPage = lazy(() => import('./pages/NewsletterPage'));
const MLAnalysisPage = lazy(() => import('./pages/MLAnalysisPage'));
const PortfolioPage = lazy(() => import('./pages/PortfolioPage'));
const AlertsPage = lazy(() => import('./pages/AlertsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));

// Layout for authenticated routes, includes Header and Sidebar
function MainLayout() {
  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-6">
          <Suspense fallback={<div className="flex justify-center items-center h-full"><LoadingSpinner size="lg" /></div>}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  );
}

function PrivateRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <div className="flex justify-center items-center h-screen"><LoadingSpinner size="lg" /></div>;
  }

  return isAuthenticated ? children : <Navigate to="/login" />;
}

function AppContent() {

  // Global error handler (can be improved with a dedicated error context)
  useEffect(() => {
    const handleError = (event) => {
      console.error('Unhandled error:', event.error);
      // Example: toast.error(event.message || 'An unexpected error occurred.');
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
