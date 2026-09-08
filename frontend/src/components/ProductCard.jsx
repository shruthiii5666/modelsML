import React from 'react';
import { useApp } from '../context/AppContext';
import { Star, Plus, Eye } from 'lucide-react';

export default function ProductCard({ product }) {
  const { addToCart, setSelectedProduct, recordEvent } = useApp();

  const handleCardClick = () => {
    recordEvent('product_click', product.product_id, `Product clicked: ${product.title}`);
    recordEvent('view', product.product_id, `Product viewed: ${product.title}`);
    setSelectedProduct(product);
  };

  const handleAddToCart = (e) => {
    e.stopPropagation();
    addToCart(product, 1);
  };

  return (
    <div className="product-card" onClick={handleCardClick}>
      {/* Product Image */}
      <div className="product-img-container">
        <img 
          src={product.image_url} 
          alt={product.title} 
          className="product-img"
          loading="lazy"
        />
      </div>

      {/* Card Body */}
      <div className="product-body">
        <div className="product-meta-row">
          <span className="product-brand">{product.brand}</span>
          <span className="product-pid">ID #{product.product_id}</span>
        </div>

        <h3 className="product-title" title={product.title}>
          {product.title}
        </h3>

        <div className="product-rating">
          <Star size={14} fill="#f59e0b" color="#f59e0b" />
          <span>{product.rating?.toFixed(1) || '4.5'}</span>
          <span className="product-rating-count">({product.reviews_count || 48})</span>
        </div>

        {/* Footer with Price and Add Button */}
        <div className="product-footer">
          <span className="product-price">${product.price?.toFixed(2)}</span>
          <button 
            className="btn btn-primary btn-sm"
            onClick={handleAddToCart}
            aria-label={`Add ${product.title} to cart`}
          >
            <Plus size={16} />
            <span>Add</span>
          </button>
        </div>
      </div>
    </div>
  );
}
