import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { X, CreditCard, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function CheckoutModal() {
  const { 
    isCheckoutOpen, 
    setIsCheckoutOpen, 
    cart, 
    cartTotal, 
    clearCart, 
    setOrderConfirmation, 
    recordEvent 
  } = useApp();

  const [shippingInfo, setShippingInfo] = useState({
    name: "Alex Morgan",
    email: "alex.morgan@example.com",
    address: "742 Evergreen Terrace",
    city: "Springfield",
    postalCode: "97477"
  });

  const [paymentMethod, setPaymentMethod] = useState("card");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (isCheckoutOpen) {
      recordEvent('checkout', 0, 'Initiated express checkout');
    }
  }, [isCheckoutOpen]);

  if (!isCheckoutOpen) return null;

  const handlePlaceOrder = (e) => {
    e.preventDefault();
    setSubmitting(true);

    const newOrderId = "ORD-" + Math.floor(100000 + Math.random() * 900000);

    setTimeout(() => {
      // Record purchase event for session tracking
      recordEvent('purchase', cart[0]?.product_id || 0, `Completed purchase for order ${newOrderId} ($${cartTotal.toFixed(2)})`);

      const orderData = {
        order_id: newOrderId,
        items: [...cart],
        total: cartTotal,
        shipping: shippingInfo,
        paymentMethod,
        date: new Date().toLocaleDateString('en-US', {
          month: 'short', day: 'numeric', year: 'numeric'
        })
      };

      setOrderConfirmation(orderData);
      clearCart();
      setSubmitting(false);
      setIsCheckoutOpen(false);
    }, 600);
  };

  return (
    <div className="modal-backdrop" onClick={() => setIsCheckoutOpen(false)}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 640 }}>
        {/* Header */}
        <div style={{
          padding: '20px 24px',
          borderBottom: '1px solid var(--border-color)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Express Checkout</h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Safe & Encrypted 256-bit SSL Checkout Simulation</p>
          </div>
          <button onClick={() => setIsCheckoutOpen(false)} style={{ color: 'var(--text-muted)', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>

        <form onSubmit={handlePlaceOrder} style={{ padding: 24 }}>
          {/* Shipping Details */}
          <div style={{ marginBottom: 20 }}>
            <h4 style={{ fontSize: '0.9rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 12 }}>
              1. Shipping Address
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div style={{ gridColumn: 'span 2' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Full Name</label>
                <input 
                  type="text" 
                  value={shippingInfo.name} 
                  onChange={(e) => setShippingInfo({...shippingInfo, name: e.target.value})}
                  style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-color)', marginTop: 4 }}
                  required 
                />
              </div>
              <div style={{ gridColumn: 'span 2' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Street Address</label>
                <input 
                  type="text" 
                  value={shippingInfo.address} 
                  onChange={(e) => setShippingInfo({...shippingInfo, address: e.target.value})}
                  style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-color)', marginTop: 4 }}
                  required 
                />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 500, color: 'var(--text-secondary)' }}>City</label>
                <input 
                  type="text" 
                  value={shippingInfo.city} 
                  onChange={(e) => setShippingInfo({...shippingInfo, city: e.target.value})}
                  style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-color)', marginTop: 4 }}
                  required 
                />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Postal Code</label>
                <input 
                  type="text" 
                  value={shippingInfo.postalCode} 
                  onChange={(e) => setShippingInfo({...shippingInfo, postalCode: e.target.value})}
                  style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-color)', marginTop: 4 }}
                  required 
                />
              </div>
            </div>
          </div>

          {/* Payment Method */}
          <div style={{ marginBottom: 24 }}>
            <h4 style={{ fontSize: '0.9rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 12 }}>
              2. Payment Method
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div 
                onClick={() => setPaymentMethod('card')}
                style={{
                  padding: 12,
                  border: `2px solid ${paymentMethod === 'card' ? 'var(--primary)' : 'var(--border-color)'}`,
                  borderRadius: 8,
                  cursor: 'pointer',
                  background: paymentMethod === 'card' ? 'var(--primary-light)' : 'transparent',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10
                }}
              >
                <CreditCard size={18} color={paymentMethod === 'card' ? 'var(--primary)' : 'var(--text-secondary)'} />
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Credit Card</span>
              </div>
              <div 
                onClick={() => setPaymentMethod('paypal')}
                style={{
                  padding: 12,
                  border: `2px solid ${paymentMethod === 'paypal' ? 'var(--primary)' : 'var(--border-color)'}`,
                  borderRadius: 8,
                  cursor: 'pointer',
                  background: paymentMethod === 'paypal' ? 'var(--primary-light)' : 'transparent',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10
                }}
              >
                <ShieldCheck size={18} color={paymentMethod === 'paypal' ? 'var(--primary)' : 'var(--text-secondary)'} />
                <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>PayPal Express</span>
              </div>
            </div>
          </div>

          {/* Order Summary Pill */}
          <div style={{
            background: 'var(--bg-muted)',
            padding: 16,
            borderRadius: 8,
            marginBottom: 24,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Total Amount Due:</span>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-heading)' }}>
                ${cartTotal.toFixed(2)}
              </div>
            </div>
            <span className="badge badge-emerald">Verified Safe Checkout</span>
          </div>

          {/* Submit */}
          <button 
            type="submit" 
            className="btn btn-primary btn-lg" 
            style={{ width: '100%' }}
            disabled={submitting}
          >
            {submitting ? 'Processing Order...' : `Place Order ($${cartTotal.toFixed(2)})`}
          </button>
        </form>
      </div>
    </div>
  );
}
