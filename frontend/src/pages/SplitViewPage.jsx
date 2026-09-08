import React from 'react';
import StorePage from './StorePage';
import DashboardPage from './DashboardPage';

export default function SplitViewPage({ searchQuery }) {
  return (
    <div className="split-view-container">
      {/* Left: Customer E-Commerce Storefront */}
      <div className="split-store-pane">
        <div style={{
          background: '#eff6ff',
          border: '1px solid #bfdbfe',
          padding: '8px 16px',
          borderRadius: 8,
          marginBottom: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <strong style={{ color: '#1e40af', fontSize: '0.85rem' }}>WEBSITE 1: Customer Storefront</strong>
            <span style={{ fontSize: '0.75rem', color: '#3b82f6', marginLeft: 8 }}>(Zero ML terminology exposed)</span>
          </div>
          <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>Customer View</span>
        </div>

        <StorePage searchQuery={searchQuery} />
      </div>

      {/* Right: Evaluator Diagnostic Dashboard */}
      <div className="split-admin-pane">
        <div style={{
          background: 'rgba(99, 102, 241, 0.15)',
          border: '1px solid rgba(99, 102, 241, 0.3)',
          padding: '8px 16px',
          borderRadius: 8,
          marginBottom: 16,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <strong style={{ color: '#a5b4fc', fontSize: '0.85rem' }}>WEBSITE 2: FYP Diagnostic Dashboard</strong>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', marginLeft: 8 }}>(Live M1 Intent & M2 Trust Shifts)</span>
          </div>
          <span className="badge badge-emerald" style={{ fontSize: '0.7rem' }}>Admin / FYP View</span>
        </div>

        <DashboardPage />
      </div>
    </div>
  );
}
