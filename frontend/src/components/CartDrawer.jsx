import React from 'react';
import { useApp } from '../context/AppContext';
import { X, Trash2, ArrowRight, ShoppingBag } from 'lucide-react';

export default function CartDrawer() {
  const { 
    isCartOpen, 
    setIsCartOpen, 
    cart, 
    removeFromCart, 
    updateQuantity, 
    cartSubtotal, 
    shippingFee, 
    estimatedTax, 
    cartTotal,
    setIsCheckoutOpen 
  } = useApp();

  if (!isCartOpen) return null;

  const handleProceedToCheckout = () => {
    setIsCartOpen(false);
    setIsCheckoutOpen(true);
  };

  return (
    <div className="modal-backdrop" onClick={() => setIsCartOpen(false)}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        {/* Drawer Header */}
        <div className="drawer-header">
          <div className="flex items-center gap-2">
            <ShoppingBag size={20} color="var(--primary)" />
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Your Shopping Cart</h3>
            <span className="badge badge-cyan">{cart.length} items</span>
          </div>
          <button 
            onClick={() => setIsCartOpen(false)}
            style={{ color: 'var(--text-muted)', cursor: 'pointer', padding: 4 }}
          >
            <X size={20} />
          </button>
        </div>

        {/* Drawer Body: Items */}
        <div className="drawer-body">
          {cart.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '64px 20px', color: 'var(--text-muted)' }}>
              <ShoppingBag size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
              <h4 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8 }}>Your cart is empty</h4>
              <p style={{ fontSize: '0.85rem', marginBottom: 20 }}>Explore our catalog and find the best verified products.</p>
              <button 
                className="btn btn-primary btn-sm"
                onClick={() => setIsCartOpen(false)}
              >
                Continue Shopping
              </button>
            </div>
          ) : (
            <div className="flex flex-col">
              {cart.map((item) => (
                <div key={item.product_id} className="cart-item">
                  <img src={item.image_url} alt={item.title} className="cart-item-img" />
                  <div className="cart-item-info">
                    <div className="flex items-center justify-between" style={{ marginBottom: 4 }}>
                      <span className="product-brand">{item.brand}</span>
                      <button 
                        onClick={() => removeFromCart(item.product_id)}
                        style={{ color: 'var(--danger)', cursor: 'pointer', opacity: 0.8 }}
                        title="Remove item"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>

                    <h4 style={{ 
                      fontSize: '0.9rem', 
                      fontWeight: 600, 
                      lineHeight: 1.25, 
                      marginBottom: 8,
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden'
                    }}>
                      {item.title}
                    </h4>

                    <div className="flex items-center justify-between">
                      <div className="qty-control">
                        <button className="qty-btn" onClick={() => updateQuantity(item.product_id, -1)}>-</button>
                        <span className="qty-value">{item.quantity}</span>
                        <button className="qty-btn" onClick={() => updateQuantity(item.product_id, 1)}>+</button>
                      </div>
                      <span style={{ fontWeight: 700, fontSize: '1rem', fontFamily: 'var(--font-heading)' }}>
                        ${(item.price * item.quantity).toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Drawer Footer: Totals & Checkout */}
        {cart.length > 0 && (
          <div className="drawer-footer">
            <div className="flex flex-col gap-2" style={{ fontSize: '0.9rem', marginBottom: 16 }}>
              <div className="flex justify-between" style={{ color: 'var(--text-secondary)' }}>
                <span>Subtotal</span>
                <span>${cartSubtotal.toFixed(2)}</span>
              </div>
              <div className="flex justify-between" style={{ color: 'var(--text-secondary)' }}>
                <span>Estimated Shipping</span>
                <span>{shippingFee === 0 ? 'FREE' : `$${shippingFee.toFixed(2)}`}</span>
              </div>
              <div className="flex justify-between" style={{ color: 'var(--text-secondary)' }}>
                <span>Estimated Sales Tax (8%)</span>
                <span>${estimatedTax.toFixed(2)}</span>
              </div>
              <div 
                className="flex justify-between" 
                style={{ 
                  fontSize: '1.15rem', 
                  fontWeight: 800, 
                  paddingTop: 8, 
                  borderTop: '1px solid var(--border-color)',
                  color: 'var(--text-primary)'
                }}
              >
                <span>Total</span>
                <span>${cartTotal.toFixed(2)}</span>
              </div>
            </div>

            <button 
              className="btn btn-primary btn-lg" 
              style={{ width: '100%' }}
              onClick={handleProceedToCheckout}
            >
              <span>Proceed to Checkout</span>
              <ArrowRight size={18} />
            </button>
          </div>
        )}
      </aside>
    </div>
  );
}
