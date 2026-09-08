import React from 'react';
import StorePage from './StorePage';
import DashboardPage from './DashboardPage';

export default function SplitViewPage({ searchQuery }) {
  return (
    <div className="split-view-container">
      {/* Left: Customer Store */}
      <div className="split-store-pane">
        <StorePage searchQuery={searchQuery} />
      </div>

      {/* Right: Monitoring Engine / Dashboard */}
      <div className="split-admin-pane">
        <DashboardPage />
      </div>
    </div>
  );
}
