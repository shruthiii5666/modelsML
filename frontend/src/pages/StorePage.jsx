import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { getCatalog, getCategories, MOCK_CATEGORIES } from '../services/api';
import ProductCard from '../components/ProductCard';
import RecommendationShelf from '../components/RecommendationShelf';
import { Sparkles, ArrowRight, ShieldCheck, Tag, Zap } from 'lucide-react';

export default function StorePage({ searchQuery }) {
  const { recordEvent } = useApp();
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState(MOCK_CATEGORIES);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);

  // Load categories from API on mount
  useEffect(() => {
    let isMounted = true;
    getCategories().then((data) => {
      if (isMounted && Array.isArray(data) && data.length > 0) {
        setCategories(data);
      }
    });
    return () => { isMounted = false; };
  }, []);

  // Track search interactions
  useEffect(() => {
    if (searchQuery && searchQuery.trim().length > 1) {
      const timer = setTimeout(() => {
        recordEvent('search', 0, `Searched for "${searchQuery}"`);
      }, 600);
      return () => clearTimeout(timer);
    }
  }, [searchQuery]);

  // Load catalog products whenever category or search filter changes
  useEffect(() => {
    let isMounted = true;
    async function load() {
      setLoading(true);
      try {
        const res = await getCatalog({
          category: selectedCategory,
          search: searchQuery,
          limit: 16
        });
        if (isMounted && res) {
          setProducts(res.products);
        }
      } catch (err) {
        console.warn("Failed to load catalog:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    load();
    return () => { isMounted = false; };
  }, [selectedCategory, searchQuery]);

  const handleCategoryClick = (cat) => {
    setSelectedCategory(cat.id);
    recordEvent('category_view', 0, `Category viewed: ${cat.label}`);
  };

  return (
    <div className="store-page">
      {/* Category Navigation Bar */}
      <nav className="category-bar">
        <div className="container">
          <div className="category-list">
            {categories.map((cat) => (
              <button
                key={cat.id}
                className={`category-tab ${selectedCategory === cat.id ? 'active' : ''}`}
                onClick={() => handleCategoryClick(cat)}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="container">
        {/* Hero Banner (Shown when not searching) */}
        {!searchQuery && selectedCategory === 'all' && (
          <section className="hero-banner">
            <div className="hero-content">
              <div className="hero-tag">
                <Sparkles size={14} />
                <span>Curated & Quality-Verified Hardware Deals</span>
              </div>
              <h1 className="hero-title">
                Next-Gen Electronics & Everyday Tech Essentials
              </h1>
              <p className="hero-desc">
                Discover verified products from world-class manufacturers. Enjoy high performance, transparent pricing, and instant 2-day delivery.
              </p>
              <div className="flex gap-3">
                <button 
                  className="btn btn-primary btn-lg"
                  onClick={() => setSelectedCategory('computers.peripherals.printer')}
                >
                  <span>Explore Printers & Tech</span>
                  <ArrowRight size={18} />
                </button>
                <button 
                  className="btn btn-secondary btn-lg"
                  onClick={() => setSelectedCategory('electronics.smartphone')}
                >
                  <span>Smartphones</span>
                </button>
              </div>
            </div>
          </section>
        )}

        {/* Section Heading */}
        <div className="flex items-center justify-between" style={{ margin: '28px 0 20px' }}>
          <div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700 }}>
              {searchQuery ? `Search Results for "${searchQuery}"` : 
                (selectedCategory === 'all' ? 'Featured Catalog Products' : 
                 MOCK_CATEGORIES.find(c => c.id === selectedCategory)?.label || 'Products')}
            </h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Showing {products.length} available items
            </p>
          </div>
        </div>

        {/* Product Grid */}
        {loading ? (
          <div className="product-grid">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="product-card" style={{ height: 340, opacity: 0.5, animation: 'pulse 1.5s infinite' }}>
                <div style={{ height: 200, background: 'var(--bg-muted)' }} />
                <div style={{ padding: 16 }}>
                  <div style={{ height: 16, background: 'var(--bg-muted)', width: '60%', marginBottom: 8, borderRadius: 4 }} />
                  <div style={{ height: 20, background: 'var(--bg-muted)', width: '90%', marginBottom: 16, borderRadius: 4 }} />
                  <div style={{ height: 24, background: 'var(--bg-muted)', width: '40%', borderRadius: 4 }} />
                </div>
              </div>
            ))}
          </div>
        ) : products.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '64px 20px', color: 'var(--text-muted)' }}>
            <Tag size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: 8 }}>No products found</h3>
            <p style={{ fontSize: '0.85rem', marginBottom: 20 }}>Try modifying your search or browsing all categories.</p>
            <button className="btn btn-primary btn-sm" onClick={() => setSelectedCategory('all')}>
              View All Products
            </button>
          </div>
        ) : (
          <div className="product-grid">
            {products.map((product) => (
              <ProductCard key={product.product_id} product={product} />
            ))}
          </div>
        )}

        {/* CRITICAL: Exactly ONE Recommendation Section */}
        <RecommendationShelf />
      </main>

      {/* Footer */}
      <footer style={{ background: 'var(--bg-surface)', borderTop: '1px solid var(--border-color)', padding: '40px 0', marginTop: 64 }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 20 }}>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.1rem', color: 'var(--primary)', marginBottom: 4 }}>ApexMart E-Commerce</div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Final Year Project: Trust-Aware E-Commerce Recommendation Using Brand Risk Analysis</p>
          </div>
          <div style={{ display: 'flex', gap: 24, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <span>Secure Checkout</span>
            <span>Verified Sellers</span>
            <span>Academic Prototype</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
