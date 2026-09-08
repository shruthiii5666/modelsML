import React, { createContext, useContext, useState, useEffect } from 'react';
import { logEvent, getSampleSessions } from '../services/api';

const AppContext = createContext();

const DEFAULT_SESSION = "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b";

export function AppProvider({ children }) {
  // Session management
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem('fyp_session_id') || DEFAULT_SESSION;
  });

  // View Mode: 'store' | 'admin' | 'split'
  const [viewMode, setViewMode] = useState(() => {
    const hash = window.location.hash;
    if (hash === '#/admin' || window.location.pathname === '/admin') return 'admin';
    if (hash === '#/split') return 'split';
    return 'store';
  });

  // Cart state
  const [cart, setCart] = useState(() => {
    try {
      const saved = localStorage.getItem('fyp_cart');
      return saved ? JSON.parse(saved) : [
        {
          product_id: 1500227,
          title: "Epson EcoTank L3150 Multi-Function Wi-Fi Printer",
          brand: "epson",
          price: 149.99,
          quantity: 1,
          image_url: "https://images.unsplash.com/photo-1612815154858-60aa4c59eaa6?w=600&auto=format&fit=crop&q=80"
        }
      ];
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
