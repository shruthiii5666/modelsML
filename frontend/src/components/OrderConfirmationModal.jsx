import React from 'react';
import { useApp } from '../context/AppContext';
import { CheckCircle2, ShoppingBag, ArrowRight } from 'lucide-react';

export default function OrderConfirmationModal() {
  const { orderConfirmation, setOrderConfirmation, setViewMode } = useApp();

  if (!orderConfirmation) return null;

  return (
    <div className="modal-backdrop" onClick={() => setOrderConfirmation(null)}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 760, padding: 32 }}>
        {/* Success Header */}
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{
            width: 64,
            height: 64,
            borderRadius: '50%',
            background: 'var(--success-bg)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 16px',
            color: 'var(--success)'
          }}>
            <CheckCircle2 size={36} />
          </div>

          <h2 style={{ fontSize: '1.75rem', fontWeight: 800, marginBottom: 6 }}>
            Order Successfully Placed!
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            Thank you for shopping with ApexMart. Your order confirmation has been sent to <strong>{orderConfirmation.shipping.email}</strong>.
          </p>
        </div>

        {/* Receipt Card */}
        <div style={{
          background: 'var(--bg-muted)',
          borderRadius: 'var(--radius-md)',
          padding: 20,
          marginBottom: 28,
          border: '1px solid var(--border-color)'
        }}>
          <div className="flex items-center justify-between" style={{ paddingBottom: 12, borderBottom: '1px solid var(--border-color)', marginBottom: 12 }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Order Number</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{orderConfirmation.order_id}</div>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Date</span>
              <div style={{ fontWeight: 600 }}>{orderConfirmation.date}</div>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>Total Paid</span>
              <div style={{ fontWeight: 800, color: 'var(--primary)', fontFamily: 'var(--font-heading)', fontSize: '1.1rem' }}>
                ${orderConfirmation.total.toFixed(2)}
              </div>
            </div>
          </div>

          {/* Ordered items */}
          <div style={{ fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: 8 }}>Items Included:</span>
            {orderConfirmation.items.map((item) => (
              <div key={item.product_id} className="flex justify-between" style={{ padding: '4px 0' }}>
                <span>{item.title} (x{item.quantity})</span>
                <span style={{ fontWeight: 600 }}>${(item.price * item.quantity).toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 12, marginBottom: 32 }}>
          <button 
            className="btn btn-primary btn-lg"
            style={{ flex: 1 }}
            onClick={() => setOrderConfirmation(null)}
          >
            <span>Continue Shopping</span>
            <ShoppingBag size={18} />
          </button>
          
          <button
            className="btn btn-secondary btn-lg"
            style={{ flex: 1 }}
            onClick={() => {
              setOrderConfirmation(null);
              setViewMode('admin');
              window.location.hash = '#/admin';
            }}
          >
            <span>View FYP Admin Telemetry</span>
            <ArrowRight size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
