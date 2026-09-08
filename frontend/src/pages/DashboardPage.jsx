import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { 
  getAdminSessionState, 
  getAdminRankComparison 
} from '../services/api';
import { 
  ShieldCheck, 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  Layers, 
  RefreshCw,
  ShoppingBag,
  Eye,
  CheckCircle
} from 'lucide-react';

export default function DashboardPage() {
  const { sessionId, lastEventTime } = useApp();

  const [sessionData, setSessionData] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load telemetry and comparison whenever session or event changes
  useEffect(() => {
    let isMounted = true;
    async function loadDashboard() {
      setLoading(true);
      try {
        const [stateRes, compRes] = await Promise.all([
          getAdminSessionState(sessionId),
          getAdminRankComparison(sessionId)
        ]);
        if (isMounted) {
          setSessionData(stateRes);
          setComparisonData(compRes);
        }
      } catch (err) {
        console.warn("Failed to load dashboard data:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadDashboard();
    return () => { isMounted = false; };
  }, [sessionId, lastEventTime]);

  if (loading && !sessionData) {
    return (
      <div className="dashboard-theme" style={{ minHeight: '100vh', padding: '60px 20px', textAlign: 'center' }}>
        <RefreshCw size={36} className="spin" style={{ margin: '0 auto 16px', color: '#6366f1' }} />
        <h3 style={{ color: '#ffffff' }}>Loading Monitoring Telemetry...</h3>
      </div>
    );
  }

  const m1 = sessionData?.model1_intent || {
    purchase_probability: 0.05
  };

  const pProb = m1.purchase_probability ?? 0;
  const pProbPct = (pProb * 100).toFixed(1);

  const m2Current = sessionData?.model2_brand_trust || {
    current_brand: sessionData?.current_brand || "epson",
    trust_score: 0.75,
    suspiciousness_score: 0.25,
    trust_category: "High Trust"
  };

  // Group events for display
  const viewedList = sessionData?.products_viewed || [];
  const cartedList = sessionData?.products_carted || sessionData?.cart_items || [];
  const purchasedList = sessionData?.products_purchased || [];
  const eventCount = sessionData?.interaction_count ?? sessionData?.timeline?.length ?? 0;

  return (
    <div className="dashboard-theme" style={{ minHeight: '100vh' }}>
      <div className="container dashboard-container">
        {/* Top Header & Current Session */}
        <div className="dashboard-header" style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#ffffff', margin: 0 }}>
              Trust-Aware Recommendation Monitoring Engine
            </h1>
          </div>

          {/* Current Session Display */}
          <div className="flex items-center gap-2">
            <span style={{ fontSize: '0.85rem', color: '#9ca3af' }}>Session:</span>
            <span className="session-id-pill" style={{ color: '#ffffff', background: '#1e293b', border: '1px solid rgba(255, 255, 255, 0.15)' }}>
              {sessionId}
            </span>
          </div>
        </div>

        {/* ============================================================
            SECTION 1: CURRENT USER SESSION
            ============================================================ */}
        <div className="dash-card" style={{ marginBottom: 24 }}>
          <div className="dash-card-header">
            <div className="dash-card-title" style={{ color: '#ffffff' }}>
              <Clock size={18} color="#6366f1" />
              <span style={{ color: '#ffffff' }}>CURRENT USER SESSION</span>
            </div>
            <span className="session-id-pill">ID: {sessionId}</span>
          </div>

          {/* Top Session Metadata Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 20 }}>
            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', display: 'block' }}>Session Start Time</span>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                {sessionData?.session_start_time || 'Just now'}
              </span>
            </div>

            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', display: 'block' }}>Session Duration</span>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff' }}>
                {sessionData?.duration_seconds ? `${Math.floor(sessionData.duration_seconds / 60)}m ${sessionData.duration_seconds % 60}s` : '0m 1s'}
              </span>
            </div>

            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', display: 'block' }}>Interaction Count</span>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#38bdf8' }}>
                {eventCount} {eventCount === 1 ? 'event' : 'events'}
              </span>
            </div>

            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', display: 'block' }}>Current Category</span>
              <span style={{ fontSize: '1rem', fontWeight: 700, color: '#a5b4fc' }}>
                {sessionData?.current_category || 'computers.peripherals.printer'}
              </span>
            </div>
          </div>

          {/* Current Activity & Current Product */}
          <div style={{ 
            background: 'rgba(255, 255, 255, 0.03)', 
            border: '1px solid var(--border-color)', 
            borderRadius: 8, 
            padding: 16, 
            marginBottom: 20,
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 16
          }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600 }}>Current Activity:</span>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#10b981', marginTop: 4 }}>
                {sessionData?.current_activity || 'Browsing Catalog'}
              </div>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600 }}>Current Product:</span>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#ffffff', marginTop: 4 }}>
                {sessionData?.current_product || 'None'}
              </div>
            </div>
          </div>

          {/* Products Viewed / Carted / Purchased Summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
            {/* Products Viewed */}
            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
                <Eye size={16} color="#38bdf8" />
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#ffffff', textTransform: 'uppercase' }}>
                  Products Viewed ({viewedList.length})
                </span>
              </div>
              <div style={{ maxHeight: 120, overflowY: 'auto', fontSize: '0.8rem', color: '#cbd5e1' }}>
                {viewedList.length === 0 ? (
                  <span style={{ color: '#64748b' }}>None viewed yet</span>
                ) : (
                  viewedList.map((p) => (
                    <div key={p.product_id} style={{ padding: '3px 0' }}>
                      • <strong>{p.brand}</strong>: {p.title || `Item #${p.product_id}`}
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Products in Cart */}
            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
                <ShoppingBag size={16} color="#10b981" />
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#ffffff', textTransform: 'uppercase' }}>
                  Products in Cart ({cartedList.length})
                </span>
              </div>
              <div style={{ maxHeight: 120, overflowY: 'auto', fontSize: '0.8rem', color: '#cbd5e1' }}>
                {cartedList.length === 0 ? (
                  <span style={{ color: '#64748b' }}>Cart is empty</span>
                ) : (
                  cartedList.map((p) => (
                    <div key={p.product_id} style={{ padding: '3px 0' }}>
                      • <strong>{p.brand}</strong>: {p.title} (x{p.quantity})
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Products Purchased */}
            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
                <CheckCircle size={16} color="#f59e0b" />
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#ffffff', textTransform: 'uppercase' }}>
                  Products Purchased ({purchasedList.length})
                </span>
              </div>
              <div style={{ maxHeight: 120, overflowY: 'auto', fontSize: '0.8rem', color: '#cbd5e1' }}>
                {purchasedList.length === 0 ? (
                  <span style={{ color: '#64748b' }}>No checkout executed yet</span>
                ) : (
                  purchasedList.map((p) => (
                    <div key={p.product_id} style={{ padding: '3px 0' }}>
                      • <strong>{p.brand}</strong>: {p.title} (x{p.quantity})
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ============================================================
            SECTION 2: SESSION TIMELINE
            ============================================================ */}
        <div className="dash-card" style={{ marginBottom: 24 }}>
          <div className="dash-card-header">
            <div className="dash-card-title" style={{ color: '#ffffff' }}>
              <Clock size={18} color="#38bdf8" />
              <span style={{ color: '#ffffff' }}>SESSION TIMELINE (Chronological User Events Stream)</span>
            </div>
            <span className="badge badge-cyan">{sessionData?.timeline?.length || 0} Events</span>
          </div>

          <div className="timeline" style={{ maxHeight: 240 }}>
            {(!sessionData?.timeline || sessionData.timeline.length === 0) ? (
              <div style={{ textAlign: 'center', padding: '24px 0', color: '#64748b', fontSize: '0.85rem' }}>
                No events recorded yet in this session. Perform interactions on the store to observe live events.
              </div>
            ) : (
              sessionData.timeline.map((evt, idx) => (
                <div key={idx} className="timeline-item">
                  <div 
                    className="timeline-bullet" 
                    style={{ 
                      background: evt.event_type === 'cart' || evt.event_type === 'cart_add' ? '#10b981' : 
                                 evt.event_type === 'purchase' ? '#f59e0b' : 
                                 evt.event_type === 'checkout' ? '#6366f1' :
                                 evt.event_type === 'search' ? '#a5b4fc' :
                                 evt.event_type === 'category_view' ? '#38bdf8' : '#3b82f6' 
                    }} 
                  />
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ fontWeight: 600, color: '#ffffff', textTransform: 'capitalize', fontSize: '0.85rem' }}>
                        {evt.event_type === 'session_start' ? 'Session Started' : 
                         evt.event_type === 'search' ? 'Search Executed' :
                         evt.event_type === 'category_view' ? 'Category Viewed' :
                         evt.event_type === 'product_view' || evt.event_type === 'view' ? 'Product Viewed' :
                         evt.event_type === 'product_click' ? 'Product Clicked' :
                         evt.event_type === 'cart' || evt.event_type === 'cart_add' ? 'Added to Cart' :
                         evt.event_type === 'quantity_change' ? 'Quantity Changed' :
                         evt.event_type === 'cart_remove' ? 'Removed from Cart' :
                         evt.event_type === 'checkout' ? 'Checkout Flow' :
                         evt.event_type === 'purchase' ? 'Purchase Completed' : evt.event_type}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: '#64748b' }}>
                        {evt.timestamp}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginTop: 2 }}>
                      {evt.detail || (evt.title ? `${evt.title} (${evt.brand})` : `Item #${evt.product_id}`)}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* ============================================================
            SECTION 3: USER INTENT (Purchase Probability) & BRAND TRUST
            ============================================================ */}
        <div className="dash-grid-2">
          {/* USER INTENT: Purchase Probability */}
          <div className="dash-card">
            <div className="dash-card-header">
              <div className="dash-card-title" style={{ color: '#ffffff' }}>
                <Activity size={18} color="#06b6d4" />
                <span style={{ color: '#ffffff' }}>Purchase Probability</span>
              </div>
            </div>

            {/* Purchase Probability Gauge */}
            <div className="gauge-container" style={{ padding: '24px 0' }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{ fontSize: '3.5rem', fontWeight: 800, color: '#06b6d4', fontFamily: 'var(--font-heading)' }}>
                  {pProbPct}%
                </span>
                <span style={{ fontSize: '1rem', color: '#9ca3af' }}>Purchase Probability</span>
              </div>

              {/* Visual Fill Bar */}
              <div className="gauge-meter-bar">
                <div className="gauge-fill" style={{ width: `${Math.min(100, pProb * 100)}%` }} />
              </div>

              <div className="flex justify-between" style={{ width: '100%', fontSize: '0.75rem', color: '#64748b' }}>
                <span>0% (Low)</span>
                <span>50%</span>
                <span>100% (High)</span>
              </div>
            </div>
          </div>

          {/* BRAND TRUST */}
          <div className="dash-card">
            <div className="dash-card-header">
              <div className="dash-card-title" style={{ color: '#ffffff' }}>
                <ShieldCheck size={18} color="#10b981" />
                <span style={{ color: '#ffffff' }}>Brand Trust</span>
              </div>
            </div>

            {/* Current Brand Highlight */}
            <div style={{ 
              background: '#1e293b', 
              borderRadius: 8, 
              padding: 16, 
              border: '1px solid rgba(255, 255, 255, 0.08)',
              marginBottom: 16
            }}>
              <div style={{ fontSize: '0.8rem', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600, marginBottom: 4 }}>
                Current Brand
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#ffffff', textTransform: 'capitalize', marginBottom: 12 }}>
                {m2Current.current_brand}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: '0.85rem' }}>
                <div>
                  <span style={{ color: '#9ca3af' }}>Trust Score: </span>
                  <strong style={{ color: m2Current.trust_score >= 0.7 ? '#10b981' : '#f59e0b' }}>
                    {m2Current.trust_score !== undefined ? m2Current.trust_score.toFixed(4) : '—'}
                  </strong>
                </div>
                <div>
                  <span style={{ color: '#9ca3af' }}>Suspiciousness: </span>
                  <strong style={{ color: '#94a3b8' }}>
                    {m2Current.suspiciousness_score !== undefined ? m2Current.suspiciousness_score.toFixed(4) : '—'}
                  </strong>
                </div>
                <div style={{ gridColumn: 'span 2' }}>
                  <span style={{ color: '#9ca3af' }}>Trust Category: </span>
                  <span style={{ color: '#cbd5e1', fontWeight: 600 }}>{m2Current.trust_category || 'Standard Trust'}</span>
                </div>
              </div>
            </div>

            {/* Scored Brands in Candidate Pool */}
            <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
              Candidate Brands Evaluated:
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#ffffff' }}>Xerox</strong>
                  <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>Low Risk</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.9107</div>
              </div>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#ffffff' }}>Canon</strong>
                  <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>Low Risk</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.7579</div>
              </div>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#ffffff' }}>HP</strong>
                  <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>Medium Risk</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.6679</div>
              </div>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#ffffff' }}>Samsung</strong>
                  <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>Medium Risk</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.7500</div>
              </div>
            </div>
          </div>
        </div>

        {/* ============================================================
            SECTION 4: FINAL RECOMMENDATION
            ============================================================ */}
        <div className="dash-card" style={{ marginBottom: 24 }}>
          <div className="dash-card-header">
            <div className="dash-card-title" style={{ color: '#ffffff' }}>
              <Layers size={18} color="#6366f1" />
              <span style={{ color: '#ffffff' }}>Final Recommendation</span>
            </div>
          </div>

          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Product</th>
                  <th>Brand</th>
                  <th>Purchase Probability</th>
                  <th>Trust</th>
                  <th>Final Score</th>
                </tr>
              </thead>
              <tbody>
                {(sessionData?.final_recommendations || comparisonData?.comparisons)
                  ?.map((item, idx) => {
                    const rank = item.rank || item.trust_aware_rank || (idx + 1);
                    const title = item.title || `Product #${item.product_id}`;
                    const pProb = item.purchase_probability !== undefined ? item.purchase_probability : item.purchase_prob;
                    return (
                      <tr key={item.product_id}>
                        <td>
                          <span style={{ 
                            width: 26, 
                            height: 26, 
                            borderRadius: '50%', 
                            background: rank === 1 ? '#6366f1' : '#1e293b', 
                            display: 'inline-flex', 
                            alignItems: 'center', 
                            justifyContent: 'center',
                            fontWeight: 700,
                            color: '#ffffff'
                          }}>
                            #{rank}
                          </span>
                        </td>
                        <td>
                          <strong style={{ color: '#ffffff' }}>{title}</strong>
                          <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>ID #{item.product_id}</div>
                        </td>
                        <td>
                          <span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{item.brand}</span>
                        </td>
                        <td>{pProb !== undefined ? `${(pProb * 100)?.toFixed(1)}%` : '—'}</td>
                        <td>
                          <span style={{ color: item.trust_score >= 0.7 ? '#10b981' : '#f59e0b', fontWeight: 700 }}>
                            {item.trust_score?.toFixed(4)}
                          </span>
                        </td>
                        <td style={{ fontWeight: 800, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                          {item.final_score?.toFixed(4)}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>

        {/* ============================================================
            SECTION 5: BEFORE VS AFTER RECOMMENDATION COMPARISON
            ============================================================ */}
        <div className="dash-card" style={{ marginBottom: 24 }}>
          <div className="dash-card-header">
            <div className="dash-card-title" style={{ color: '#ffffff' }}>
              <span style={{ color: '#ffffff' }}>Purchase Only Ordering Recommendation versus Trust-Aware Recommendation</span>
            </div>
          </div>

          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Product</th>
                  <th>Brand</th>
                  <th>Risk Level</th>
                  <th>Before Rank</th>
                  <th>After Rank</th>
                  <th>Rank Shift</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData?.comparisons?.map((item) => (
                  <tr key={item.product_id}>
                    <td>
                      <strong style={{ color: '#ffffff' }}>{item.title || `Product #${item.product_id}`}</strong>
                      <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>ID #{item.product_id}</div>
                    </td>
                    <td><span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{item.brand}</span></td>
                    <td>
                      <span className={`badge ${item.risk_level === 'Low' ? 'badge-emerald' : 'badge-amber'}`}>
                        {item.risk_level} Risk
                      </span>
                    </td>
                    <td style={{ color: '#9ca3af' }}>Rank #{item.purchase_only_rank}</td>
                    <td><strong style={{ color: '#ffffff' }}>Rank #{item.trust_aware_rank}</strong></td>
                    <td>
                      {item.rank_shift > 0 ? (
                        <span className="shift-promoted">
                          <TrendingUp size={16} />
                          +{item.rank_shift} (PROMOTED)
                        </span>
                      ) : item.rank_shift < 0 ? (
                        <span className="shift-demoted">
                          <TrendingDown size={16} />
                          {item.rank_shift} (DEMOTED)
                        </span>
                      ) : (
                        <span style={{ color: '#94a3b8' }}>0 (UNCHANGED)</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
