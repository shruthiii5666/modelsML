import React, { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { getRecommendations } from '../services/api';
import ProductCard from './ProductCard';
import { Sparkles, RefreshCw } from 'lucide-react';

export default function RecommendationShelf() {
  const { sessionId, lastEventTime } = useApp();
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch recommendations whenever session or user interaction updates
  useEffect(() => {
    let isMounted = true;
    async function loadRecs() {
      setLoading(true);
      try {
        const res = await getRecommendations(sessionId, 4);
        if (isMounted && res && res.recommendations) {
          setRecommendations(res.recommendations);
        }
      } catch (err) {
        console.warn("Failed to load recommendations:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadRecs();
    return () => { isMounted = false; };
  }, [sessionId, lastEventTime]);

  if (!loading && recommendations.length === 0) {
    return null;
  }

  return (
    <section className="recommendations-shelf">
      <div className="shelf-header">
        <div className="shelf-title-wrap">
          <div style={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            background: 'var(--primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white'
          }}>
            <Sparkles size={16} />
          </div>
          <h2 className="shelf-title">Recommended for You</h2>
          <span className="shelf-badge">Personalized</span>
        </div>

        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            <RefreshCw size={14} className="spin" />
            <span>Updating suggestions...</span>
          </div>
        )}
      </div>

      <div className="product-grid" style={{ marginBottom: 0 }}>
        {recommendations.map((product) => (
          <ProductCard key={product.product_id} product={product} />
        ))}
      </div>
    </section>
  );
}
