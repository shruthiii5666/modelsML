import React from 'react';
import { useApp } from '../context/AppContext';
import { 
  ShoppingBag, 
  Search, 
  LayoutDashboard, 
  Store, 
  Columns, 
  Sparkles,
  RefreshCw 
} from 'lucide-react';

export default function Navbar({ onSearchChange, searchQuery }) {
  const { 
    cartItemCount, 
    setIsCartOpen, 
    viewMode, 
    setViewMode, 
    sessionId,
    createNewSession 
  } = useApp();

  return (
    <header className="navbar">
      <div className="container nav-content">
        {/* Brand Logo */}
        <div className="flex items-center gap-4">
          <div 
            className="brand-logo" 
            style={{ cursor: 'pointer' }}
            onClick={() => setViewMode('store')}
          >
            <div style={{
              width: 36,
              height: 36,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #2563eb, #6366f1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff'
            }}>
              <ShoppingBag size={20} />
            </div>
            <span>ApexMart</span>
            <span className="brand-badge">FYP DEMO</span>
          </div>

          {/* Quick Evaluator Session Display */}
          <div className="flex items-center gap-2" style={{ marginLeft: 8 }}>
            <span style={{ 
              fontSize: '0.75rem', 
              color: 'var(--text-muted)', 
              fontFamily: 'var(--font-mono)' 
            }}>
              Session: {sessionId ? `${sessionId.substring(0, 8)}...` : 'Active'}
            </span>
            <button 
              onClick={createNewSession}
              title="Generate fresh session"
              style={{ color: 'var(--text-muted)', padding: 4 }}
            >
              <RefreshCw size={12} />
            </button>
          </div>
        </div>

        {/* Search Bar (Only shown or prominent in Store & Split view) */}
        {viewMode !== 'admin' && (
          <div className="nav-search">
            <Search size={18} className="nav-search-icon" />
            <input
              type="text"
              placeholder="Search products, brands (e.g. Epson, Xerox, Apple)..."
              value={searchQuery || ''}
              onChange={(e) => onSearchChange && onSearchChange(e.target.value)}
            />
          </div>
        )}

        {/* Navigation & Actions */}
        <div className="nav-links">
          {/* View Mode Switcher Pill */}
          <div className="view-mode-pill">
            <button
              className={`view-mode-btn ${viewMode === 'store' ? 'active' : ''}`}
              onClick={() => {
                setViewMode('store');
                window.location.hash = '#/';
              }}
              title="Customer E-Commerce Storefront (Website 1)"
            >
              <Store size={14} style={{ display: 'inline', marginRight: 4 }} />
              Store
            </button>
            <button
              className={`view-mode-btn ${viewMode === 'admin' ? 'active' : ''}`}
              onClick={() => {
                setViewMode('admin');
                window.location.hash = '#/admin';
              }}
              title="Admin / FYP Diagnostic Dashboard (Website 2)"
            >
              <LayoutDashboard size={14} style={{ display: 'inline', marginRight: 4 }} />
              Dashboard
            </button>
            <button
              className={`view-mode-btn ${viewMode === 'split' ? 'active' : ''}`}
              onClick={() => {
                setViewMode('split');
                window.location.hash = '#/split';
              }}
              title="Dual-Pane Evaluator Split View"
            >
              <Columns size={14} style={{ display: 'inline', marginRight: 4 }} />
              Split
            </button>
          </div>

          {/* Cart Button */}
          <button 
            className="cart-btn"
            onClick={() => setIsCartOpen(true)}
            aria-label="View shopping cart"
          >
            <ShoppingBag size={20} />
            {cartItemCount > 0 && (
              <span className="cart-badge">{cartItemCount}</span>
            )}
          </button>
        </div>
      </div>
    </header>
  );
}
