import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { X, Star, ShoppingBag, Check, ShieldCheck, Truck } from 'lucide-react';

export default function ProductDetailModal() {
  const { 
    selectedProduct, 
    setSelectedProduct, 
    addToCart, 
    setIsCheckoutOpen 
  } = useApp();
  const [qty, setQty] = useState(1);

  if (!selectedProduct) return null;

  const handleClose = () => setSelectedProduct(null);

  const handleAddToCart = () => {
    addToCart(selectedProduct, qty);
    handleClose();
  };

  const handleBuyNow = () => {
    addToCart(selectedProduct, qty);
    handleClose();
    setIsCheckoutOpen(true);
  };

  return (
    <div className="modal-backdrop" onClick={handleClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 800 }}>
        {/* Close Button */}
        <button 
          onClick={handleClose}
          style={{
            position: 'absolute',
            top: 16,
            right: 16,
            background: 'var(--bg-muted)',
            border: 'none',
            borderRadius: '50%',
            width: 36,
            height: 36,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            zIndex: 10
          }}
        >
          <X size={20} />
        </button>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 32, padding: 32 }}>
          {/* Left: Product Media */}
          <div style={{
            background: 'var(--bg-muted)',
            borderRadius: 'var(--radius-md)',
            padding: 24,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <img 
              src={selectedProduct.image_url} 
              alt={selectedProduct.title}
              style={{ maxWidth: '100%', maxHeight: 320, objectFit: 'contain' }}
            />
          </div>

          {/* Right: Details & Purchase */}
          <div className="flex flex-col">
            <div className="product-meta-row" style={{ marginBottom: 4 }}>
              <span className="product-brand" style={{ fontSize: '0.85rem' }}>{selectedProduct.brand}</span>
              <span className="product-pid">Item #{selectedProduct.product_id}</span>
            </div>

            <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: 12, lineHeight: 1.3 }}>
              {selectedProduct.title}
            </h2>

            <div className="product-rating" style={{ marginBottom: 16 }}>
              <Star size={16} fill="#f59e0b" color="#f59e0b" />
              <span style={{ fontSize: '0.9rem' }}>{selectedProduct.rating?.toFixed(1) || '4.6'}</span>
              <span className="product-rating-count">({selectedProduct.reviews_count || 128} verified customer ratings)</span>
            </div>

            <div style={{ 
              fontSize: '1.75rem', 
              fontWeight: 800, 
              color: 'var(--primary)', 
              marginBottom: 16,
              fontFamily: 'var(--font-heading)'
            }}>
              ${selectedProduct.price?.toFixed(2)}
            </div>

            {/* Guarantees */}
            <div style={{ 
              display: 'flex', 
              gap: 16, 
              padding: '12px 0', 
              borderTop: '1px solid var(--border-color)',
              borderBottom: '1px solid var(--border-color)',
              marginBottom: 20,
              fontSize: '0.8rem',
              color: 'var(--text-secondary)'
            }}>
              <div className="flex items-center gap-2">
                <Truck size={16} color="#2563eb" />
                <span>Free 2-Day Shipping</span>
              </div>
              <div className="flex items-center gap-2">
                <ShieldCheck size={16} color="#10b981" />
                <span>Quality Verified</span>
              </div>
            </div>

            {/* Technical Specifications */}
            {selectedProduct.specs && (
              <div style={{ marginBottom: 20 }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 8 }}>
                  Specifications
                </h4>
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '1fr 1fr', 
                  gap: '6px 16px', 
                  fontSize: '0.85rem' 
                }}>
                  {Object.entries(selectedProduct.specs).map(([key, val]) => (
                    <div key={key}>
                      <span style={{ color: 'var(--text-muted)' }}>{key}: </span>
                      <span style={{ fontWeight: 500 }}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Quantity and Actions */}
            <div style={{ marginTop: 'auto', display: 'flex', gap: 12, alignItems: 'center' }}>
              <div className="qty-control">
                <button className="qty-btn" onClick={() => setQty(Math.max(1, qty - 1))}>-</button>
                <span className="qty-value">{qty}</span>
                <button className="qty-btn" onClick={() => setQty(qty + 1)}>+</button>
              </div>

              <button 
                className="btn btn-primary"
                style={{ flex: 1 }}
                onClick={handleAddToCart}
              >
                <ShoppingBag size={18} />
                <span>Add to Cart</span>
              </button>

              <button 
                className="btn btn-secondary"
                style={{ flex: 1 }}
                onClick={handleBuyNow}
              >
                <span>Buy Now</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
