import React, { useState, useEffect } from 'react';
import { AppProvider, useApp } from './context/AppContext';
import Navbar from './components/Navbar';
import StorePage from './pages/StorePage';
import DashboardPage from './pages/DashboardPage';
import SplitViewPage from './pages/SplitViewPage';
import ProductDetailModal from './components/ProductDetailModal';
import CartDrawer from './components/CartDrawer';
import CheckoutModal from './components/CheckoutModal';
import OrderConfirmationModal from './components/OrderConfirmationModal';

function MainLayout() {
  const { viewMode, setViewMode, toast } = useApp();
  const [searchQuery, setSearchQuery] = useState('');

  // Sync hash with viewMode
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash;
      if (hash === '#/admin') setViewMode('admin');
      else if (hash === '#/split') setViewMode('split');
      else setViewMode('store');
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [setViewMode]);

  return (
    <div className={viewMode === 'admin' ? 'dashboard-theme' : ''}>
      {/* Navigation Header */}
      <Navbar onSearchChange={setSearchQuery} searchQuery={searchQuery} />

      {/* Main View Router */}
      {viewMode === 'store' && <StorePage searchQuery={searchQuery} />}
      {viewMode === 'admin' && <DashboardPage />}
      {viewMode === 'split' && <SplitViewPage searchQuery={searchQuery} />}

      {/* Global Modals & Drawers */}
      <ProductDetailModal />
      <CartDrawer />
      <CheckoutModal />
      <OrderConfirmationModal />

      {/* Toast Notifications */}
      {toast && (
        <div style={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          background: toast.type === 'success' ? '#10b981' : '#1e293b',
          color: 'white',
          padding: '12px 20px',
          borderRadius: 8,
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.2)',
          fontSize: '0.9rem',
          fontWeight: 600,
          zIndex: 200,
          animation: 'fadeIn 0.2s ease-out'
        }}>
          {toast.message}
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <MainLayout />
    </AppProvider>
  );
}
