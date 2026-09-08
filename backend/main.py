import uuid
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.service import (
    DATA_STORE,
    SESSION_MANAGER,
    generate_recommendations_for_session,
    get_dashboard_telemetry
)

# Initialize FastAPI App
app = FastAPI(
    title="Trust-Aware E-Commerce Recommendation API",
    description="Thin integration layer between React frontends and existing ML models.",
    version="1.0.0"
)

# Configure CORS to allow React frontends on any local port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC REQUEST SCHEMAS
# ============================================================

class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = None

class EventLogRequest(BaseModel):
    session_id: Optional[str] = None
    event_type: str = Field(..., description="view, search, category_view, product_click, cart, quantity_change, cart_remove, checkout, purchase")
    product_id: Optional[int] = None
    details: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "models": {
            "model1_user_intent": DATA_STORE.m1_loaded,
            "model2_brand_trust": DATA_STORE.trust_df is not None,
            "product_catalog": DATA_STORE.catalog_df is not None,
            "recommendation_engine": True
        }
    }


# ============================================================
# PRODUCTS & CATALOG ENDPOINTS
# ============================================================

def _filter_catalog(
    page: int = 1,
    limit: int = 12,
    category: Optional[str] = None,
    search: Optional[str] = None
) -> dict:
    if DATA_STORE.catalog_df is None or DATA_STORE.catalog_df.empty:
        raise HTTPException(status_code=503, detail="Product catalog not loaded")

    df = DATA_STORE.catalog_df

    if category and isinstance(category, str) and category.lower() != "all":
        df = df[df["category_code"].str.lower() == category.lower()]

    if search and isinstance(search, str):
        q = search.lower().strip()
        mask = df["brand"].str.lower().str.contains(q, na=False) | df["category_code"].str.lower().str.contains(q, na=False)
        df = df[mask]

    total = len(df)
    start_idx = (page - 1) * limit
    page_df = df.iloc[start_idx : start_idx + limit]

    products = [DATA_STORE.format_product(int(row["product_id"]), row.to_dict()) for _, row in page_df.iterrows()]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "products": products
    }


@app.get("/api/products", tags=["Products"])
@app.get("/api/catalog", tags=["Products"])
def get_products(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(12, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Category filter"),
    search: Optional[str] = Query(None, description="Search query")
):
    """Browse catalog products with optional category and search filtering."""
    return _filter_catalog(page=page, limit=limit, category=category, search=search)


@app.get("/api/products/{product_id}", tags=["Products"])
def get_product_by_id(product_id: int = Path(..., description="Unique product ID")):
    """Get complete product details."""
    if DATA_STORE.catalog_dict is None or product_id not in DATA_STORE.catalog_dict:
        raise HTTPException(status_code=404, detail=f"Product #{product_id} not found in catalog")

    return DATA_STORE.format_product(product_id)


@app.get("/api/categories", tags=["Products"])
def get_categories():
    """Returns top product categories with item counts."""
    categories = [
        {"id": "all", "label": "All Products", "count": len(DATA_STORE.catalog_df) if DATA_STORE.catalog_df is not None else 0},
        {"id": "computers.peripherals.printer", "label": "Computers & Printers", "count": len(DATA_STORE.category_products.get("computers.peripherals.printer", []))},
        {"id": "electronics.smartphone", "label": "Smartphones", "count": len(DATA_STORE.category_products.get("electronics.smartphone", []))},
        {"id": "electronics.audio.headphone", "label": "Audio & Headphones", "count": len(DATA_STORE.category_products.get("electronics.audio.headphone", []))},
        {"id": "appliances.environment.water_heater", "label": "Appliances", "count": len(DATA_STORE.category_products.get("appliances.environment.water_heater", []))},
        {"id": "kids.toys", "label": "Kids & Toys", "count": len(DATA_STORE.category_products.get("kids.toys", []))}
    ]
    return categories


@app.get("/api/search", tags=["Products"])
def search_products(
    q: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(12, ge=1, le=50)
):
    """Search products by title or brand."""
    return _filter_catalog(page=1, limit=limit, search=q)


# ============================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================

@app.post("/api/session", tags=["Session"])
def create_or_initialize_session(payload: Optional[SessionCreateRequest] = Body(None)):
    """Initializes a new session or retrieves an existing one."""
    sid = payload.session_id if payload and payload.session_id else str(uuid.uuid4())
    sess = SESSION_MANAGER.get_or_create(sid)
    return sess.to_dict()


@app.post("/api/session/{session_id}/event", tags=["Session"])
def log_session_event_path(
    session_id: str = Path(..., description="Active session ID"),
    payload: EventLogRequest = Body(...)
):
    """
    Records customer interaction event and triggers Model 1 inference.
    Supports: product_view, search, category_view, product_click, add_to_cart,
    remove_from_cart, quantity_change, checkout, purchase.
    """
    return SESSION_MANAGER.record_event(
        session_id=session_id,
        event_type=payload.event_type,
        product_id=payload.product_id,
        details=payload.details,
        metadata=payload.metadata
    )


@app.post("/api/events", tags=["Session"])
def log_session_event_body(
    payload: EventLogRequest = Body(...)
):
    """Records customer interaction event using session_id from body."""
    sid = payload.session_id or "default_session"
    return SESSION_MANAGER.record_event(
        session_id=sid,
        event_type=payload.event_type,
        product_id=payload.product_id,
        details=payload.details,
        metadata=payload.metadata
    )


@app.get("/api/session/{session_id}", tags=["Session"])
def get_session_by_id(session_id: str = Path(..., description="Active session ID")):
    """Returns complete real-time session telemetry."""
    sess = SESSION_MANAGER.get_or_create(session_id)
    return sess.to_dict()


# ============================================================
# RECOMMENDATION ENDPOINT
# (Strictly ONE customer recommendation section, zero ML terminology)
# ============================================================

@app.get("/api/recommendations/{session_id}", tags=["Recommendation"])
def get_recommendations_by_path(
    session_id: str = Path(..., description="Active session ID"),
    limit: int = Query(8, ge=1, le=20)
):
    """
    Returns final customer-safe recommendations (strictly zero ML terminology).
    Executes:
    1. Identify current session
    2. Obtain Model 1 intent probability
    3. Obtain Model 2 brand trust & risk
    4. Call existing compute_trust_aware_scores() from recommendation/ranking.py
    5. Return final ranked recommendations
    """
    recommendations = generate_recommendations_for_session(session_id, limit=limit)
    return {
        "session_id": session_id,
        "recommendations": recommendations
    }


@app.get("/api/recommendations", tags=["Recommendation"])
def get_recommendations_by_query(
    session_id: Optional[str] = Query(None, description="Active session ID"),
    limit: int = Query(8, ge=1, le=20)
):
    """Query parameter fallback for recommendations."""
    sid = session_id or "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b"
    recommendations = generate_recommendations_for_session(sid, limit=limit)
    return {
        "session_id": sid,
        "recommendations": recommendations
    }


# ============================================================
# ADMIN / FYP EVALUATION DASHBOARD ENDPOINTS
# ============================================================

@app.get("/api/dashboard/{session_id}", tags=["Dashboard"])
def get_dashboard_by_path(
    session_id: str = Path(..., description="Active session ID")
):
    """
    Returns full telemetry for Website 2 (Admin / FYP Dashboard):
    - Session information
    - Chronological timeline
    - Model 1 User Intent
    - Model 2 Brand Trust
    - Final Trust-Aware Recommendations table
    - Before (Purchase-Only) vs. Final (Trust-Aware) Rank Shifts table
    """
    return get_dashboard_telemetry(session_id)


@app.get("/api/dashboard", tags=["Dashboard"])
@app.get("/api/admin/session-state", tags=["Dashboard"])
def get_dashboard_by_query(
    session_id: Optional[str] = Query(None, description="Active session ID")
):
    """Query parameter fallback for dashboard telemetry."""
    sid = session_id or "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b"
    return get_dashboard_telemetry(sid)


@app.get("/api/admin/rank-comparison", tags=["Dashboard"])
def get_rank_comparison(session_id: Optional[str] = Query(None)):
    """Returns Before (Purchase-Only) vs. Final (Trust-Aware) comparison table."""
    sid = session_id or "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b"
    telemetry = get_dashboard_telemetry(sid)
    return telemetry["before_vs_final"]


@app.get("/api/admin/sample-sessions", tags=["Dashboard"])
def get_sample_sessions():
    """Returns benchmark thesis sessions."""
    return [
        {
            "session_id": "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
            "title": "Printer Shopper (High Intent, High Trust Promotions)",
            "category": "computers.peripherals.printer",
            "default_intent": 0.8105,
            "description": "Evaluates Epson views, Xerox high-trust promotion (+8 rank boost), and HP penalty demotion."
        },
        {
            "session_id": "2bb8e316-9856-441e-a858-5723f916b456",
            "title": "Kids Toys (Significant Unknown Brand Demotions)",
            "category": "kids.toys",
            "default_intent": 0.1200,
            "description": "Safe verified brands (Simba, CubicFun) promoted, while unknown-brand toy dropped by 8 ranks."
        },
        {
            "session_id": "5a5350f2-c5d7-4230-8fec-3ac32c134ab3",
            "title": "Smartphone Search (Xiaomi vs Samsung Re-ordering)",
            "category": "electronics.smartphone",
            "default_intent": 0.4500,
            "description": "Evaluates behavioral consistency between established consumer electronics manufacturers."
        }
    ]


@app.get("/api/admin/metrics", tags=["Dashboard"])
def get_admin_metrics():
    """Returns verified academic benchmark metrics."""
    return {
        "models": [
            {
                "name": "Trust-Aware (Proposed)",
                "hit_rate_10": "94.10%",
                "ndcg_10": "0.6078",
                "average_trust_10": "0.6991",
                "high_risk_exposure_10": "0.00%"
            }
        ]
    }
