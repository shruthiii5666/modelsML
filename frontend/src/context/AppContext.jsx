import React, { createContext, useContext, useState, useEffect } from 'react';
import { logEvent, getSampleSessions } from '../services/api';

const AppContext = createContext();

function getInitialSessionId() {
  const saved = localStorage.getItem('fyp_session_id');
  const evalSessions = [
    "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
    "2bb8e316-9856-441e-a858-5723f916b456",
    "5a5350f2-c5d7-4230-8fec-3ac32c134ab3"
  ];
  if (saved && !evalSessions.includes(saved)) {
    return saved;
  }
  const newId = 'live_' + Math.random().toString(36).substring(2, 9);
  localStorage.setItem('fyp_session_id', newId);
  return newId;
}

export function AppProvider({ children }) {
  // Session management - starts with fresh live customer session
  const [sessionId, setSessionId] = useState(getInitialSessionId);

  // View Mode: 'store' | 'admin' | 'split'
  const [viewMode, setViewMode] = useState(() => {
    const hash = window.location.hash;
    if (hash === '#/admin' || window.location.pathname === '/admin') return 'admin';
    if (hash === '#/split') return 'split';
    return 'store';
  });

  // Cart state - starts empty for a live session unless restored from storage
  const [cart, setCart] = useState(() => {
    try {
      const saved = localStorage.getItem('fyp_cart');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  // UI state
  const [isCartOpen, setIsCartOpen] = useState(false);
  const [isCheckoutOpen, setIsCheckoutOpen] = useState(false);
  const [orderConfirmation, setOrderConfirmation] = useState(null);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [toast, setToast] = useState(null);
  const [lastEventTime, setLastEventTime] = useState(Date.now());

  // Save session and cart to localStorage
  useEffect(() => {
    localStorage.setItem('fyp_session_id', sessionId);
  }, [sessionId]);

  useEffect(() => {
    localStorage.setItem('fyp_cart', JSON.stringify(cart));
  }, [cart]);

  // Show temporary toast notification
  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  // Record an interaction event
  const recordEvent = async (eventType, productId = 0, details = '', metadata = {}) => {
    try {
      await logEvent({ 
        session_id: sessionId, 
        event_type: eventType, 
        product_id: productId, 
        details, 
        metadata 
      });
      setLastEventTime(Date.now());
    } catch (err) {
      console.warn("Event dispatch error:", err);
    }
  };

  // Cart actions
  const addToCart = (product, qty = 1) => {
    setCart(prev => {
      const existing = prev.find(item => item.product_id === product.product_id);
      if (existing) {
        return prev.map(item =>
          item.product_id === product.product_id
            ? { ...item, quantity: item.quantity + qty }
            : item
        );
      }
      return [...prev, {
        product_id: product.product_id,
        title: product.title,
        brand: product.brand,
        price: product.price,
        image_url: product.image_url,
        quantity: qty
      }];
    });

    recordEvent('cart', product.product_id, `Added to cart: ${product.title}`, { qty });
    showToast(`Added ${product.title} to cart!`, 'success');
  };

  const removeFromCart = (productId) => {
    const item = cart.find(i => i.product_id === productId);
    setCart(prev => prev.filter(i => i.product_id !== productId));
    recordEvent('cart_remove', productId, `Removed from cart: ${item ? item.title : '#' + productId}`);
    showToast('Item removed from cart', 'info');
  };

  const updateQuantity = (productId, delta) => {
    const item = cart.find(i => i.product_id === productId);
    setCart(prev => prev.map(i => {
      if (i.product_id === productId) {
        const newQty = i.quantity + delta;
        return newQty > 0 ? { ...i, quantity: newQty } : null;
      }
      return i;
    }).filter(Boolean));
    recordEvent('quantity_change', productId, `Quantity changed for ${item ? item.title : '#' + productId} (${delta > 0 ? '+1' : '-1'})`);
  };

  const clearCart = () => {
    setCart([]);
  };

  const createNewSession = () => {
    const newId = 'sess_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now().toString().slice(-4);
    setSessionId(newId);
    setCart([]);
    recordEvent('session_start', 0);
    showToast(`Started fresh session: ${newId}`, 'info');
  };

  // Calculate cart totals
  const cartSubtotal = cart.reduce((acc, item) => acc + item.price * item.quantity, 0);
  const cartItemCount = cart.reduce((acc, item) => acc + item.quantity, 0);
  const shippingFee = cartSubtotal > 50 ? 0 : 9.99;
  const estimatedTax = cartSubtotal * 0.08;
  const cartTotal = cartSubtotal + (cartSubtotal > 0 ? shippingFee : 0) + estimatedTax;

  return (
    <AppContext.Provider
      value={{
        sessionId,
        setSessionId,
        viewMode,
        setViewMode,
        cart,
        addToCart,
        removeFromCart,
        updateQuantity,
        clearCart,
        cartSubtotal,
        cartItemCount,
        shippingFee,
        estimatedTax,
        cartTotal,
        isCartOpen,
        setIsCartOpen,
        isCheckoutOpen,
        setIsCheckoutOpen,
        orderConfirmation,
        setOrderConfirmation,
        selectedProduct,
        setSelectedProduct,
        toast,
        showToast,
        recordEvent,
        lastEventTime,
        createNewSession
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error("useApp must be used within AppProvider");
  return context;
}
