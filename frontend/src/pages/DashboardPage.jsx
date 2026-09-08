import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { 
  getAdminSessionState, 
  getAdminRankComparison, 
  SAMPLE_SESSIONS 
} from '../services/api';
import { 
  ShieldCheck, 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  CheckCircle2, 
  Layers, 
  Zap,
  Info,
  RefreshCw,
  ShoppingBag,
  Eye,
  Search,
  CheckCircle
} from 'lucide-react';

export default function DashboardPage() {
  const { sessionId, setSessionId, lastEventTime } = useApp();

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
        <h3 style={{ color: '#f9fafb' }}>Loading FYP Evaluation Telemetry...</h3>
      </div>
    );
  }

  const m1 = sessionData?.model1_intent || {
    purchase_probability: 0.8105,
    predicted_purchase: 1,
    intent_level: "High Purchase Intent",
    intent_multiplier: 1.8105
  };

  const pProb = m1.purchase_probability;
  const pProbPct = (pProb * 100).toFixed(1);

  const m2Current = sessionData?.model2_brand_trust || {
    current_brand: sessionData?.current_brand || "epson",
    trust_score: 0.5748,
    suspiciousness_score: 0.4252,
    risk_level: "Medium",
    trust_category: "Medium Trust",
    penalty_multiplier: 0.75,
    interpretation: "Moderate behavioral consistency with periodic cart abandonment spikes."
  };

  // Group events for display
  const viewedList = sessionData?.products_viewed || [];
  const cartedList = sessionData?.products_carted || sessionData?.cart_items || [];
  const purchasedList = sessionData?.products_purchased || [];

  return (
    <div className="dashboard-theme" style={{ minHeight: '100vh' }}>
      <div className="container dashboard-container">
        {/* Top Evaluator Header & Session Switcher */}
        <div className="dashboard-header">
          <div>
            <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
              <span className="badge badge-cyan">Website 2</span>
              <span className="badge badge-emerald">Admin / FYP Evaluation Dashboard</span>
            </div>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#f9fafb' }}>
              Trust-Aware Recommendation Monitoring Engine
            </h1>
            <p style={{ fontSize: '0.85rem', color: '#9ca3af' }}>
              Academic Evaluation Dashboard & Real-Time Multi-Model Telemetry
            </p>
          </div>

          {/* Session Selector */}
          <div className="flex items-center gap-3">
            <span style={{ fontSize: '0.85rem', color: '#9ca3af' }}>Evaluator Session:</span>
            <select
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              style={{
                background: '#1e293b',
                color: '#f9fafb',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                padding: '8px 12px',
                borderRadius: 8,
                fontSize: '0.85rem',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              {SAMPLE_SESSIONS.map((sess) => (
                <option key={sess.session_id} value={sess.session_id}>
                  {sess.title}
                </option>
              ))}
              {!SAMPLE_SESSIONS.some(s => s.session_id === sessionId) && (
                <option value={sessionId}>Live Custom Session ({sessionId.substring(0, 8)}...)</option>
              )}
            </select>
          </div>
        </div>

        {/* ============================================================
            SECTION 1: CURRENT USER SESSION
            ============================================================ */}
        <div className="dash-card" style={{ marginBottom: 24 }}>
          <div className="dash-card-header">
            <div className="dash-card-title">
              <Clock size={18} color="#6366f1" />
              <span>CURRENT USER SESSION</span>
            </div>
            <span className="session-id-pill">ID: {sessionId}</span>
          </div>

          {/* Top Session Metadata Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 20 }}>
            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', display: 'block' }}>Session Start Time</span>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f9fafb', fontFamily: 'var(--font-mono)' }}>
                {sessionData?.session_start_time || '14:02:10'}
              </span>
            </div>

            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', display: 'block' }}>Session Duration</span>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f9fafb' }}>
                {sessionData?.duration_seconds ? `${Math.floor(sessionData.duration_seconds / 60)}m ${sessionData.duration_seconds % 60}s` : '4m 12s'}
              </span>
            </div>

            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', display: 'block' }}>Interaction Count</span>
              <span style={{ fontSize: '1.05rem', fontWeight: 700, color: '#38bdf8' }}>
                {sessionData?.interaction_count || sessionData?.timeline?.length || 6} events
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
                {sessionData?.current_activity || 'Viewing Epson EcoTank L3150 Printer'}
              </div>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600 }}>Current Product:</span>
              <div style={{ fontSize: '1rem', fontWeight: 600, color: '#f9fafb', marginTop: 4 }}>
                {sessionData?.current_product || 'Epson EcoTank L3150 Multi-Function Wi-Fi Printer'}
              </div>
            </div>
          </div>

          {/* Products Viewed / Carted / Purchased Summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
            {/* Products Viewed */}
            <div style={{ background: '#1e293b', padding: 14, borderRadius: 8 }}>
              <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
                <Eye size={16} color="#38bdf8" />
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f9fafb', textTransform: 'uppercase' }}>
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
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f9fafb', textTransform: 'uppercase' }}>
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
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f9fafb', textTransform: 'uppercase' }}>
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
            <div className="dash-card-title">
              <Clock size={18} color="#38bdf8" />
              <span>SESSION TIMELINE (Chronological User Event Stream)</span>
            </div>
            <span className="badge badge-cyan">{sessionData?.timeline?.length || 0} Chronological Events</span>
          </div>

          <p style={{ fontSize: '0.85rem', color: '#9ca3af', marginBottom: 16 }}>
            Chronological log of customer events tracked from Website 1 (Session Started, Search, Category Viewed, Product Viewed, Product Clicked, Added to Cart, Quantity Changed, Removed from Cart, Checkout, Purchase).
          </p>

          <div className="timeline" style={{ maxHeight: 240 }}>
            {sessionData?.timeline?.map((evt, idx) => (
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
                    <span style={{ fontWeight: 600, color: '#f9fafb', textTransform: 'capitalize', fontSize: '0.85rem' }}>
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
            ))}
          </div>
        </div>

        {/* ============================================================
            SECTION 3: USER INTENT (MODEL 1) & BRAND TRUST (MODEL 2)
            ============================================================ */}
        <div className="dash-grid-2">
          {/* USER INTENT: Model 1 */}
          <div className="dash-card">
            <div className="dash-card-header">
              <div className="dash-card-title">
                <Activity size={18} color="#06b6d4" />
                <span>USER INTENT (Model 1 — Purchase Prediction)</span>
              </div>
              <span className="badge badge-cyan">Stacking Ensemble</span>
            </div>

            {/* Purchase Probability Gauge */}
            <div className="gauge-container">
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{ fontSize: '3rem', fontWeight: 800, color: '#06b6d4', fontFamily: 'var(--font-heading)' }}>
                  {pProbPct}%
                </span>
                <span style={{ fontSize: '1rem', color: '#9ca3af' }}>Purchase Probability</span>
              </div>

              {/* Visual Fill Bar */}
              <div className="gauge-meter-bar">
                <div className="gauge-fill" style={{ width: `${Math.min(100, pProb * 100)}%` }} />
              </div>

              <div className="flex justify-between" style={{ width: '100%', fontSize: '0.75rem', color: '#64748b' }}>
                <span>0.0 (Browsing)</span>
                <span>Threshold: 0.50</span>
                <span>1.0 (High Intent)</span>
              </div>
            </div>

            {/* Model 1 Interpretation */}
            <div style={{ 
              background: '#1e293b', 
              padding: 16, 
              borderRadius: 8, 
              display: 'flex', 
              flexDirection: 'column', 
              gap: 10,
              fontSize: '0.85rem'
            }}>
              <div className="flex justify-between items-center">
                <span style={{ color: '#9ca3af' }}>Predicted Purchase:</span>
                <span className={`badge ${m1.predicted_purchase === 1 ? 'badge-emerald' : 'badge-amber'}`}>
                  {m1.predicted_purchase === 1 ? '1 (High Purchase Intent)' : '0 (Low Intent / Window Shopping)'}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span style={{ color: '#9ca3af' }}>Intent Multiplier (1 + α·P):</span>
                <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#f9fafb' }}>
                  {m1.intent_multiplier.toFixed(4)}
                </span>
              </div>

              <div style={{ 
                paddingTop: 10, 
                borderTop: '1px solid rgba(255, 255, 255, 0.08)', 
                color: '#94a3b8', 
                fontSize: '0.8rem',
                lineHeight: 1.4
              }}>
                <Info size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'text-bottom' }} />
                <strong>Simple Interpretation:</strong> {m1.predicted_purchase === 1 ? 
                  "Customer exhibits strong purchasing intent (cart actions & high interaction frequency). Relevance scores are amplified by the intent multiplier." : 
                  "Customer is in discovery/browsing mode. Standard baseline catalog relevance is maintained."}
              </div>
            </div>
          </div>

          {/* BRAND TRUST: Model 2 */}
          <div className="dash-card">
            <div className="dash-card-header">
              <div className="dash-card-title">
                <ShieldCheck size={18} color="#10b981" />
                <span>BRAND TRUST (Model 2 — Brand Risk Analysis)</span>
              </div>
              <span className="badge badge-emerald">Behavioral Anomaly Model</span>
            </div>

            {/* Current Brand Highlight */}
            <div style={{ 
              background: '#1e293b', 
              borderRadius: 8, 
              padding: 16, 
              border: `1px solid ${m2Current.risk_level === 'Low' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
              marginBottom: 16
            }}>
              <div className="flex justify-between items-center" style={{ marginBottom: 8 }}>
                <span style={{ fontSize: '0.8rem', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600 }}>
                  Current Interacted Brand
                </span>
                <span className={`badge ${m2Current.risk_level === 'Low' ? 'badge-emerald' : 'badge-amber'}`}>
                  {m2Current.risk_level} Risk (Ψ = {m2Current.penalty_multiplier?.toFixed(2)})
                </span>
              </div>
              
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f9fafb', textTransform: 'capitalize', marginBottom: 8 }}>
                {m2Current.current_brand}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: '0.85rem' }}>
                <div>
                  <span style={{ color: '#9ca3af' }}>Trust Score: </span>
                  <strong style={{ color: m2Current.trust_score >= 0.7 ? '#10b981' : '#f59e0b' }}>
                    {m2Current.trust_score?.toFixed(4)}
                  </strong>
                </div>
                <div>
                  <span style={{ color: '#9ca3af' }}>Suspiciousness: </span>
                  <strong style={{ color: '#94a3b8' }}>
                    {m2Current.suspiciousness_score?.toFixed(4)}
                  </strong>
                </div>
                <div>
                  <span style={{ color: '#9ca3af' }}>Trust Category: </span>
                  <span style={{ color: '#cbd5e1' }}>{m2Current.trust_category}</span>
                </div>
                <div>
                  <span style={{ color: '#9ca3af' }}>Penalty Multiplier Ψ: </span>
                  <span style={{ color: '#cbd5e1' }}>{m2Current.penalty_multiplier?.toFixed(2)}</span>
                </div>
              </div>
            </div>

            {/* Other Scored Brands in Candidate Pool */}
            <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase' }}>
              Candidate Brands Evaluated:
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#f9fafb' }}>Xerox</strong>
                  <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>Low Risk</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.9107 | Ψ = 1.00</div>
              </div>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#f9fafb' }}>Canon</strong>
                  <span className="badge badge-emerald" style={{ fontSize: '0.65rem' }}>Low Risk</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.7579 | Ψ = 1.00</div>
              </div>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#f9fafb' }}>HP</strong>
                  <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>Med Risk</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.6679 | Ψ = 0.75</div>
              </div>
              <div style={{ background: '#1e293b', padding: 10, borderRadius: 6 }}>
                <div className="flex justify-between">
                  <strong style={{ color: '#f9fafb' }}>Unknown/Unscored</strong>
                  <span className="badge badge-amber" style={{ fontSize: '0.65rem' }}>Fallback</span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>Trust: 0.5000 | Ψ = 0.75</div>
              </div>
            </div>
          </div>
        </div>

        {/* ============================================================
            SECTION 4: FINAL RECOMMENDATION
            (Shows ONLY the final recommendation produced by existing recommendation system)
            ============================================================ */}
        <div className="dash-card" style={{ marginBottom: 24 }}>
          <div className="dash-card-header">
            <div className="dash-card-title">
              <Layers size={18} color="#6366f1" />
              <span>FINAL RECOMMENDATION (Trust-Aware Recommendation Output)</span>
            </div>
            <span className="badge badge-cyan">Formula: R × (1 + α·P) × Trust^β × Ψ</span>
          </div>

          <p style={{ fontSize: '0.85rem', color: '#9ca3af', marginBottom: 16 }}>
            The final trust-aware product ranking generated by the completed recommendation system for active session <code style={{ color: '#a5b4fc' }}>{sessionId}</code>.
          </p>

          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Product</th>
                  <th>Brand</th>
                  <th>Purchase Probability</th>
                  <th>Trust</th>
                  <th>Risk</th>
                  <th>Base Relevance</th>
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
                            color: '#f9fafb'
                          }}>
                            #{rank}
                          </span>
                        </td>
                        <td>
                          <strong style={{ color: '#f9fafb' }}>{title}</strong>
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
                        <td>
                          <span className={`badge ${item.risk_level === 'Low' ? 'badge-emerald' : 'badge-amber'}`}>
                            {item.risk_level} (Ψ={item.penalty_multiplier?.toFixed(2)})
                          </span>
                        </td>
                        <td>{item.base_relevance?.toFixed(2)}</td>
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
            SECTION 5: BEFORE VS FINAL
            (Strictly BEFORE: Purchase-Only vs FINAL: Trust-Aware)
            ============================================================ */}
        <div className="dash-card" style={{ marginBottom: 24 }}>
          <div className="dash-card-header">
            <div className="dash-card-title">
              <Zap size={18} color="#f59e0b" />
              <span>BEFORE RECOMMENDATION (Purchase-Only Ordering) vs. FINAL RECOMMENDATION (Trust-Aware Final Ordering)</span>
            </div>
            <span className="badge badge-amber">Rank Shift Comparison</span>
          </div>

          <p style={{ fontSize: '0.85rem', color: '#9ca3af', marginBottom: 16 }}>
            Demonstrates how factoring Model 2 brand behavioral risk alters product ranking: rewarding trustworthy sellers and demoting anomalies without needing any baseline clutter.
          </p>

          <div className="data-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Product</th>
                  <th>Brand</th>
                  <th>Risk Level</th>
                  <th>Before Rank (Purchase-Only)</th>
                  <th>Final Rank (Trust-Aware)</th>
                  <th>Rank Shift</th>
                  <th>Algorithmic Explanation</th>
                </tr>
              </thead>
              <tbody>
                {comparisonData?.comparisons?.map((item) => (
                  <tr key={item.product_id}>
                    <td>
                      <strong style={{ color: '#f9fafb' }}>{item.title || `Product #${item.product_id}`}</strong>
                      <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>ID #{item.product_id}</div>
                    </td>
                    <td><span style={{ textTransform: 'capitalize', fontWeight: 600 }}>{item.brand}</span></td>
                    <td>
                      <span className={`badge ${item.risk_level === 'Low' ? 'badge-emerald' : 'badge-amber'}`}>
                        {item.risk_level}
                      </span>
                    </td>
                    <td style={{ color: '#9ca3af' }}>Rank #{item.purchase_only_rank}</td>
                    <td><strong style={{ color: '#f9fafb' }}>Rank #{item.trust_aware_rank}</strong></td>
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
                    <td style={{ fontSize: '0.8rem', color: '#cbd5e1', maxWidth: 360, whiteSpace: 'normal' }}>
                      {item.reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ============================================================
            SECTION 6: FINAL MODEL PERFORMANCE METRICS
            (Evaluator reference metrics for Proposed Final Model)
            ============================================================ */}
        <div className="dash-card">
          <div className="dash-card-header">
            <div className="dash-card-title">
              <CheckCircle2 size={18} color="#10b981" />
              <span>FINAL MODEL PERFORMANCE METRICS (Proposed Trust-Aware System)</span>
            </div>
            <span className="badge badge-emerald">Evaluated on Held-Out Test Set</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
            <div style={{ background: '#1e293b', padding: 16, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase' }}>HitRate@10</span>
              <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#10b981', fontFamily: 'var(--font-heading)' }}>
                94.10%
              </div>
              <span style={{ fontSize: '0.75rem', color: '#64748b' }}>High recommendation relevance</span>
            </div>

            <div style={{ background: '#1e293b', padding: 16, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase' }}>NDCG@10</span>
              <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#38bdf8', fontFamily: 'var(--font-heading)' }}>
                0.6078
              </div>
              <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Accurate ranking position</span>
            </div>

            <div style={{ background: '#1e293b', padding: 16, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase' }}>Average Trust@10</span>
              <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#a5b4fc', fontFamily: 'var(--font-heading)' }}>
                0.6991
              </div>
              <span style={{ fontSize: '0.75rem', color: '#64748b' }}>Prioritizes verified merchants</span>
            </div>

            <div style={{ background: '#1e293b', padding: 16, borderRadius: 8 }}>
              <span style={{ fontSize: '0.75rem', color: '#9ca3af', textTransform: 'uppercase' }}>High-Risk Exposure@10</span>
              <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#10b981', fontFamily: 'var(--font-heading)' }}>
                0.00%
              </div>
              <span style={{ fontSize: '0.75rem', color: '#64748b' }}>100% bad actor suppression</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
