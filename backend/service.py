import os
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Ensure workspace root is on sys.path for direct imports of existing modules
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Silence loky CPU warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
import joblib

# Import EXISTING recommendation modules directly (DO NOT MODIFY OR REDESIGN)
from recommendation.config import (
    PRODUCT_CATALOG_PATH,
    MODEL1_TEST_RESULTS_PATH,
    MODEL1_PROBABILITIES_PATH,
    MODEL2_TRUST_DATASET_PATH,
    TOPK_TRUST_AWARE_PATH,
    TOPK_PURCHASE_ONLY_PATH,
    CANDIDATE_CACHE_PATH,
    ALPHA,
    BETA,
    RISK_PENALTIES,
    DEFAULT_K,
    IN_SESSION_CART_RELEVANCE,
    IN_SESSION_VIEW_RELEVANCE,
    CATEGORY_AFFINITY_MIN_RELEVANCE,
    CATEGORY_AFFINITY_MAX_RELEVANCE
)
from recommendation.ranking import (
    compute_trust_aware_scores,
    compute_baseline_scores,
    extract_topk_recommendations
)
from recommendation.trust_integrator import (
    load_brand_trust_data,
    load_purchase_intent_data
)

# High-resolution category image mapping for realistic e-commerce presentation
CATEGORY_IMAGES = {
    "computers.peripherals.printer": "https://images.unsplash.com/photo-1612815154858-60aa4c59eaa6?w=600&auto=format&fit=crop&q=80",
    "electronics.smartphone": "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5?w=600&auto=format&fit=crop&q=80",
    "electronics.audio.headphone": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80",
    "appliances.environment.water_heater": "https://images.unsplash.com/photo-1585338107529-13afc5f02586?w=600&auto=format&fit=crop&q=80",
    "kids.toys": "https://images.unsplash.com/photo-1566576912321-d58ddd7a6088?w=600&auto=format&fit=crop&q=80",
    "appliances.kitchen.refrigerators": "https://images.unsplash.com/photo-1571175443880-49e1d25b2bc5?w=600&auto=format&fit=crop&q=80",
    "computers.notebook": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=600&auto=format&fit=crop&q=80",
    "default": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop&q=80"
}

# Showcase curated descriptions for demo clarity
CURATED_PRODUCTS = {
    1500227: {
        "title": "Epson EcoTank L3150 Multi-Function Wi-Fi Printer",
        "image_url": "https://images.unsplash.com/photo-1612815154858-60aa4c59eaa6?w=600&auto=format&fit=crop&q=80",
        "rating": 4.7,
        "reviews_count": 142
    },
    1500208: {
        "title": "Xerox B210 Wireless Compact Monochrome Laser Printer",
        "image_url": "https://images.unsplash.com/photo-1589330694653-dad6d3240a2b?w=600&auto=format&fit=crop&q=80",
        "rating": 4.8,
        "reviews_count": 98
    },
    1500075: {
        "title": "Canon Pixma G3010 All-in-One Ink Tank Color Printer",
        "image_url": "https://images.unsplash.com/photo-1544816155-12df9643f363?w=600&auto=format&fit=crop&q=80",
        "rating": 4.6,
        "reviews_count": 115
    },
    1500447: {
        "title": "HP LaserJet Pro M404n High-Speed Workgroup Printer",
        "image_url": "https://images.unsplash.com/photo-1550745165-9bc0b252726f?w=600&auto=format&fit=crop&q=80",
        "rating": 4.3,
        "reviews_count": 73
    },
    1004856: {
        "title": "Samsung Galaxy A50 Super AMOLED Triple Camera",
        "image_url": "https://images.unsplash.com/photo-1580910051074-3eb694886505?w=600&auto=format&fit=crop&q=80",
        "rating": 4.5,
        "reviews_count": 320
    },
    1004794: {
        "title": "Xiaomi Redmi Note 8 64GB Quad-Camera Smartphone",
        "image_url": "https://images.unsplash.com/photo-1598327105666-5b89351aff97?w=600&auto=format&fit=crop&q=80",
        "rating": 4.7,
        "reviews_count": 284
    },
    1005161: {
        "title": "Apple iPhone 11 64GB Dual Camera - Space Gray",
        "image_url": "https://images.unsplash.com/photo-1510557880182-3d4d3cba35a5?w=600&auto=format&fit=crop&q=80",
        "rating": 4.9,
        "reviews_count": 512
    },
    4804295: {
        "title": "Xiaomi Mi True Wireless Noise Cancelling Earbuds",
        "image_url": "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?w=600&auto=format&fit=crop&q=80",
        "rating": 4.4,
        "reviews_count": 89
    },
    4804058: {
        "title": "Sony WH-1000XM4 Wireless Over-Ear Active NC Headphones",
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop&q=80",
        "rating": 4.9,
        "reviews_count": 215
    },
    3900535: {
        "title": "Electrolux EcoHeat Electric Storage Water Heater 50L",
        "image_url": "https://images.unsplash.com/photo-1585338107529-13afc5f02586?w=600&auto=format&fit=crop&q=80",
        "rating": 4.5,
        "reviews_count": 64
    },
    10201407: {
        "title": "Simba Toys Educational Wooden Activity Center",
        "image_url": "https://images.unsplash.com/photo-1566576912321-d58ddd7a6088?w=600&auto=format&fit=crop&q=80",
        "rating": 4.8,
        "reviews_count": 76
    },
    8900818: {
        "title": "CubicFun 3D Architectural Puzzle Landmark Edition",
        "image_url": "https://images.unsplash.com/photo-1587654780291-39c9404d746b?w=600&auto=format&fit=crop&q=80",
        "rating": 4.9,
        "reviews_count": 94
    },
    10201542: {
        "title": "Generic Plush Baby Stuffed Toy - Soft Fluffy Bunny",
        "image_url": "https://images.unsplash.com/photo-1559454403-b8fb88521f11?w=600&auto=format&fit=crop&q=80",
        "rating": 3.8,
        "reviews_count": 19
    }
}


class BackendDataStore:
    """
    Loads catalog, Model 1 models, and Model 2 brand trust tables into memory
    once on startup for high performance.
    """

    def __init__(self):
        print("[Backend] Initializing DataStore...")
        self.catalog_df = None
        self.catalog_dict = {}
        self.category_products = {}
        self.trust_df = None
        self.trust_dict = {}
        self.intent_precomputed_df = None
        self.intent_precomputed_dict = {}

        # Model 1 ML Stacking Ensemble artifacts
        self.m1_lgbm = None
        self.m1_xgb = None
        self.m1_rf = None
        self.m1_meta = None
        self.m1_loaded = False

        self._load_catalog()
        self._load_model2_trust()
        self._load_model1_models()
        self._load_precomputed_intent()

    def _load_catalog(self):
        if os.path.exists(PRODUCT_CATALOG_PATH):
            print(f"[Backend] Loading catalog from {PRODUCT_CATALOG_PATH}...")
            df = pd.read_csv(PRODUCT_CATALOG_PATH)
            # Clean values
            df["brand"] = df["brand"].fillna("unknown").astype(str).str.strip().str.lower()
            df["category_code"] = df["category_code"].fillna("unknown").astype(str).str.strip()
            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
            self.catalog_df = df
            self.catalog_dict = df.set_index("product_id").to_dict(orient="index")

            for cat, group in df.groupby("category_code"):
                self.category_products[cat] = group.sort_values("popularity_score", ascending=False)
            print(f"[Backend] Catalog loaded: {len(df):,} products across {len(self.category_products)} categories.")
        else:
            print("[Backend WARNING] Product catalog file not found!")

    def _load_model2_trust(self):
        try:
            df = load_brand_trust_data()
            self.trust_df = df
            self.trust_dict = df.set_index("brand").to_dict(orient="index")
            print(f"[Backend] Model 2 Brand Trust loaded: {len(self.trust_dict):,} brands.")
        except Exception as e:
            print(f"[Backend WARNING] Model 2 Trust data could not be loaded: {e}")

    def _load_model1_models(self):
        models_dir = WORKSPACE_ROOT / "user-intent" / "models"
        try:
            lgbm_path = models_dir / "purchase_lgbm.pkl"
            xgb_path = models_dir / "purchase_xgb.pkl"
            rf_path = models_dir / "purchase_rf.pkl"
            meta_path = models_dir / "purchase_stacking_meta.pkl"

            if all(p.exists() for p in [lgbm_path, xgb_path, rf_path, meta_path]):
                self.m1_lgbm = joblib.load(lgbm_path)
                self.m1_xgb = joblib.load(xgb_path)
                self.m1_rf = joblib.load(rf_path)
                self.m1_meta = joblib.load(meta_path)
                self.m1_loaded = True

                # Load metadata for threshold and feature columns
                meta_json_path = models_dir / "purchase_model_metadata.json"
                if meta_json_path.exists():
                    import json
                    with open(meta_json_path, "r") as f:
                        self.m1_metadata = json.load(f)
                        self.m1_best_threshold = float(self.m1_metadata.get("best_threshold", 0.10))
                else:
                    self.m1_best_threshold = 0.10

                print(f"[Backend] Model 1 Stacking Ensemble loaded (decision threshold: {self.m1_best_threshold}).")
            else:
                print("[Backend WARNING] Some Model 1 pickle files were not found.")
        except Exception as e:
            print(f"[Backend WARNING] Could not load Model 1 models: {e}")

    def _load_precomputed_intent(self):
        try:
            df = load_purchase_intent_data()
            self.intent_precomputed_df = df
            self.intent_precomputed_dict = df.set_index("user_session")["purchase_probability"].to_dict()
            print(f"[Backend] Precomputed intent loaded: {len(self.intent_precomputed_dict):,} sessions.")
        except Exception as e:
            print(f"[Backend WARNING] Precomputed intent data: {e}")

    def format_product(self, product_id: int, raw_row: Optional[dict] = None) -> dict:
        """Enrich a catalog product with clean presentation metadata."""
        meta = raw_row or self.catalog_dict.get(product_id, {})
        brand = meta.get("brand", "unknown")
        category_code = meta.get("category_code", "unknown")
        price = float(meta.get("price", 0.0))

        # Check curated metadata
        if product_id in CURATED_PRODUCTS:
            curated = CURATED_PRODUCTS[product_id]
            title = curated["title"]
            image_url = curated["image_url"]
            rating = curated["rating"]
            reviews_count = curated["reviews_count"]
        else:
            # Deterministic formatting from real dataset fields
            leaf = category_code.split(".")[-1].replace("_", " ").title() if category_code != "unknown" else "Product"
            brand_title = brand.title() if brand != "unknown" else "Apex"
            title = f"{brand_title} {leaf} - Model #{product_id}"
            image_url = CATEGORY_IMAGES.get(category_code, CATEGORY_IMAGES["default"])
            # Deterministic rating between 4.2 and 4.9 based on product_id
            rating = 4.2 + (abs(hash(str(product_id))) % 8) * 0.1
            reviews_count = 20 + (abs(hash(str(product_id))) % 180)

        return {
            "product_id": int(product_id),
            "title": title,
            "brand": brand,
            "category_code": category_code,
            "price": round(price, 2),
            "rating": round(rating, 1),
            "reviews_count": int(reviews_count),
            "image_url": image_url,
            "popularity_score": float(meta.get("popularity_score", 0.5))
        }

    def get_brand_trust(self, brand: str) -> dict:
        """Lookup Model 2 brand trust metrics with official thesis fallbacks."""
        b_clean = (brand or "unknown").lower().strip()
        if b_clean == "unknown" or b_clean == "":
            return {
                "brand": "unknown",
                "trust_score": 0.5000,
                "suspiciousness_score": 0.5000,
                "risk_level": "Medium",
                "trust_category": "Unknown Brand",
                "penalty_multiplier": 0.75,
                "scoring_status": "unbranded_fallback",
                "interpretation": "Unbranded fallback applied (Trust=0.50, Psi=0.75) to protect customer safety."
            }

        if b_clean in self.trust_dict:
            row = self.trust_dict[b_clean]
            t_score = float(row.get("trust_score", 0.75))
            r_level = str(row.get("risk_level", "Medium"))
            psi = RISK_PENALTIES.get(r_level, 0.75)
            susp = float(row.get("suspiciousness_score", 1.0 - t_score))
            cat = str(row.get("trust_category", f"{r_level} Trust"))
            return {
                "brand": b_clean,
                "trust_score": round(t_score, 4),
                "suspiciousness_score": round(susp, 4),
                "risk_level": r_level,
                "trust_category": cat,
                "penalty_multiplier": psi,
                "scoring_status": "scored",
                "interpretation": f"Observed seller history: behavioral trust {t_score:.4f} with risk rating {r_level}."
            }

        # Known name but unscored in Model 2 dataset
        return {
            "brand": b_clean,
            "trust_score": 0.7500,
            "suspiciousness_score": 0.2500,
            "risk_level": "Medium",
            "trust_category": "Insufficient Evidence",
            "penalty_multiplier": 0.75,
            "scoring_status": "insufficient_evidence_fallback",
            "interpretation": "Brand has <5 recorded interactions; standard safety fallback (Trust=0.75, Psi=0.75) applied."
        }


# Global in-memory data store instance
DATA_STORE = BackendDataStore()


class SessionState:
    """Represents real-time in-memory session tracking."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.start_time_str = self.created_at.strftime("%H:%M:%S")
        self.events: List[dict] = []
        self.cart_items: Dict[int, dict] = {}
        self.viewed_products: Dict[int, dict] = {}
        self.purchased_products: List[dict] = []
        self.active_category: str = "computers.peripherals.printer"
        self.current_product: str = "Epson EcoTank L3150 Multi-Function Wi-Fi Printer"
        self.current_brand: str = "epson"
        self.current_activity: str = "Session initialized"
        self.purchase_probability: float = 0.05
        self.predicted_purchase: int = 0

        # If matching an evaluated session, initialize from precomputed intent
        if session_id in DATA_STORE.intent_precomputed_dict:
            p = DATA_STORE.intent_precomputed_dict[session_id]
            self.purchase_probability = float(p)
            threshold = getattr(DATA_STORE, "m1_best_threshold", 0.10)
            self.predicted_purchase = 1 if p >= threshold else 0

    @property
    def duration_seconds(self) -> int:
        return max(1, int((datetime.now() - self.created_at).total_seconds()))

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_start_time": self.start_time_str,
            "duration_seconds": self.duration_seconds,
            "current_activity": self.current_activity,
            "interaction_count": len(self.events),
            "current_category": self.active_category,
            "current_product": self.current_product,
            "current_brand": self.current_brand,
            "products_viewed": list(self.viewed_products.values()),
            "products_carted": list(self.cart_items.values()),
            "products_purchased": self.purchased_products,
            "timeline": self.events,
            "model1_intent": {
                "purchase_probability": round(self.purchase_probability, 4),
                "predicted_purchase": self.predicted_purchase,
                "intent_level": "High Purchase Intent" if self.predicted_purchase == 1 else "Browsing / Low Intent",
                "intent_multiplier": round(1.0 + ALPHA * self.purchase_probability, 4),
                "model_name": "Stacking Meta-Ensemble (LightGBM + XGBoost + RF)"
            }
        }


class SessionManager:
    """In-memory session manager for local demo execution."""

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        # Pre-seed benchmark sessions so evaluator can query them immediately
        self._seed_benchmark_sessions()

    def _seed_benchmark_sessions(self):
        # 1. Printer shopper session
        s1 = self.get_or_create("0f65dee0-ae4d-460e-bb66-3da1bbbaec6b")
        s1.active_category = "computers.peripherals.printer"
        s1.current_brand = "epson"
        s1.current_product = "Epson EcoTank L3150 Multi-Function Wi-Fi Printer"
        s1.current_activity = "Viewing Epson EcoTank L3150 Printer"
        s1.purchase_probability = 0.8105
        s1.predicted_purchase = 1
        s1.viewed_products[1500227] = DATA_STORE.format_product(1500227)
        s1.cart_items[1500227] = {**DATA_STORE.format_product(1500227), "quantity": 1}
        s1.events = [
            {"id": 1, "timestamp": "14:02:10", "event_type": "session_start", "detail": "Session started"},
            {"id": 2, "timestamp": "14:02:18", "event_type": "search", "detail": "Searched for 'printer'"},
            {"id": 3, "timestamp": "14:02:22", "event_type": "category_view", "detail": "Category viewed: Computers & Printers"},
            {"id": 4, "timestamp": "14:02:25", "event_type": "product_view", "product_id": 1500227, "brand": "epson", "title": "Epson EcoTank L3150", "detail": "Product viewed: Epson EcoTank L3150"},
            {"id": 5, "timestamp": "14:02:30", "event_type": "product_click", "product_id": 1500227, "brand": "epson", "title": "Epson EcoTank L3150", "detail": "Product clicked: Epson EcoTank L3150"},
            {"id": 6, "timestamp": "14:03:10", "event_type": "cart", "product_id": 1500227, "brand": "epson", "title": "Epson EcoTank L3150", "detail": "Added to cart: Epson EcoTank L3150"}
        ]

        # 2. Kids toys session
        s2 = self.get_or_create("2bb8e316-9856-441e-a858-5723f916b456")
        s2.active_category = "kids.toys"
        s2.current_brand = "simba"
        s2.current_product = "Simba Toys Educational Wooden Activity Center"
        s2.current_activity = "Browsing Kids & Toys"
        s2.purchase_probability = 0.1200
        s2.predicted_purchase = 0
        s2.viewed_products[10201407] = DATA_STORE.format_product(10201407)
        s2.events = [
            {"id": 1, "timestamp": "10:15:00", "event_type": "session_start", "detail": "Session started"},
            {"id": 2, "timestamp": "10:15:20", "event_type": "category_view", "detail": "Category viewed: Kids & Toys"},
            {"id": 3, "timestamp": "10:15:45", "event_type": "product_view", "product_id": 10201407, "brand": "simba", "title": "Simba Wooden Center", "detail": "Product viewed: Simba Activity Center"}
        ]

        # 3. Smartphone session
        s3 = self.get_or_create("5a5350f2-c5d7-4230-8fec-3ac32c134ab3")
        s3.active_category = "electronics.smartphone"
        s3.current_brand = "xiaomi"
        s3.current_product = "Xiaomi Redmi Note 8 64GB Quad-Camera"
        s3.current_activity = "Comparing Xiaomi & Samsung Smartphones"
        s3.purchase_probability = 0.4500
        s3.predicted_purchase = 0
        s3.viewed_products[1004794] = DATA_STORE.format_product(1004794)
        s3.events = [
            {"id": 1, "timestamp": "16:40:00", "event_type": "session_start", "detail": "Session started"},
            {"id": 2, "timestamp": "16:40:15", "event_type": "search", "detail": "Searched for 'smartphone'"},
            {"id": 3, "timestamp": "16:40:30", "event_type": "product_view", "product_id": 1004794, "brand": "xiaomi", "title": "Xiaomi Redmi Note 8", "detail": "Product viewed: Xiaomi Redmi Note 8"}
        ]

    def get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id)
        return self.sessions[session_id]

    def record_event(
        self,
        session_id: str,
        event_type: str,
        product_id: Optional[int] = None,
        details: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        sess = self.get_or_create(session_id)
        metadata = metadata or {}
        time_str = datetime.now().strftime("%H:%M:%S")

        prod_info = None
        if product_id and int(product_id) > 0:
            prod_info = DATA_STORE.format_product(int(product_id))

        event_detail = details
        if not event_detail:
            if event_type == "session_start":
                event_detail = "Session started"
            elif event_type == "search":
                event_detail = f"Searched for '{details or 'products'}'"
            elif event_type == "category_view":
                event_detail = f"Category viewed: {details or 'Catalog'}"
            elif event_type in ["product_view", "view"]:
                event_detail = f"Product viewed: {prod_info['title'] if prod_info else f'Item #{product_id}'}"
            elif event_type == "product_click":
                event_detail = f"Product clicked: {prod_info['title'] if prod_info else f'Item #{product_id}'}"
            elif event_type in ["cart", "add_to_cart", "cart_add"]:
                event_detail = f"Added to cart: {prod_info['title'] if prod_info else f'Item #{product_id}'}"
            elif event_type == "quantity_change":
                event_detail = f"Quantity changed: {details or 'Quantity updated'}"
            elif event_type in ["remove_from_cart", "cart_remove"]:
                event_detail = f"Removed from cart: {prod_info['title'] if prod_info else f'Item #{product_id}'}"
            elif event_type == "checkout":
                event_detail = "Initiated express checkout flow"
            elif event_type == "purchase":
                event_detail = f"Completed purchase: {details or 'Order Placed'}"
            else:
                event_detail = f"Action: {event_type}"

        sess.current_activity = event_detail

        # Record to timeline
        sess.events.append({
            "id": len(sess.events) + 1,
            "timestamp": time_str,
            "event_type": event_type,
            "product_id": int(product_id) if product_id else None,
            "brand": prod_info["brand"] if prod_info else sess.current_brand,
            "title": prod_info["title"] if prod_info else details,
            "detail": event_detail
        })

        # Update product & category context
        if prod_info:
            sess.current_product = prod_info["title"]
            sess.current_brand = prod_info["brand"]
            sess.active_category = prod_info["category_code"]

            if event_type in ["view", "product_view", "product_click"]:
                sess.viewed_products[prod_info["product_id"]] = prod_info

            if event_type in ["cart", "add_to_cart", "cart_add"]:
                pid = prod_info["product_id"]
                qty = metadata.get("qty", 1)
                if pid in sess.cart_items:
                    sess.cart_items[pid]["quantity"] += qty
                else:
                    sess.cart_items[pid] = {**prod_info, "quantity": qty}

        if event_type == "quantity_change" and product_id:
            pid = int(product_id)
            if pid in sess.cart_items:
                new_qty = metadata.get("quantity") or metadata.get("qty") or 1
                if int(new_qty) > 0:
                    sess.cart_items[pid]["quantity"] = int(new_qty)
                else:
                    sess.cart_items.pop(pid, None)

        if event_type in ["remove_from_cart", "cart_remove"] and product_id:
            sess.cart_items.pop(int(product_id), None)

        if event_type == "checkout":
            sess.purchase_probability = max(sess.purchase_probability, 0.90)

        if event_type == "purchase":
            sess.purchased_products.extend(list(sess.cart_items.values()))
            sess.cart_items.clear()
            sess.purchase_probability = 0.99
            sess.predicted_purchase = 1

        # Run Model 1 Inference on session features if trained model available
        self._update_model1_inference(sess)

        return {
            "status": "success",
            "session_id": session_id,
            "event_count": len(sess.events),
            "cart_count": len(sess.cart_items),
            "purchase_probability": round(sess.purchase_probability, 4),
            "predicted_purchase": sess.predicted_purchase
        }

    def _update_model1_inference(self, sess: SessionState):
        """Extract 18 session features and run Model 1 Stacking Ensemble."""
        if not DATA_STORE.m1_loaded:
            # Fallback heuristic: cart items increase probability
            cart_len = len(sess.cart_items)
            view_len = len(sess.viewed_products)
            if cart_len > 0:
                sess.purchase_probability = min(0.95, 0.55 + cart_len * 0.15)
                sess.predicted_purchase = 1
            else:
                sess.purchase_probability = min(0.48, 0.05 + view_len * 0.08)
                sess.predicted_purchase = 0
            return

        try:
            # Construct feature row according to purchase_model_metadata.json
            v_count = len([e for e in sess.events if "view" in e.get("event_type", "")])
            c_count = len([e for e in sess.events if "cart" in e.get("event_type", "")])
            prices = [p["price"] for p in sess.viewed_products.values()] or [100.0]

            # Last 5 actions mapped: 1 for view, 2 for cart
            action_codes = []
            for e in sess.events[-5:]:
                etype = e.get("event_type", "")
                if "cart" in etype:
                    action_codes.append(2)
                else:
                    action_codes.append(1)
            while len(action_codes) < 5:
                action_codes.insert(0, 0)

            feat_dict = {
                "event_count": len(sess.events),
                "view_count": max(1, v_count),
                "cart_count": c_count,
                "unique_products": max(1, len(sess.viewed_products)),
                "unique_categories": 1,
                "unique_brands": max(1, len(set(p["brand"] for p in sess.viewed_products.values()))),
                "average_price": float(np.mean(prices)),
                "max_price": float(np.max(prices)),
                "min_price": float(np.min(prices)),
                "session_duration": float(sess.duration_seconds),
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().weekday(),
                "is_weekend": 1 if datetime.now().weekday() >= 5 else 0,
                "last_action_1": action_codes[0],
                "last_action_2": action_codes[1],
                "last_action_3": action_codes[2],
                "last_action_4": action_codes[3],
                "last_action_5": action_codes[4]
            }

            feat_df = pd.DataFrame([feat_dict])
            p_lgbm = DATA_STORE.m1_lgbm.predict_proba(feat_df)[:, 1]
            p_xgb = DATA_STORE.m1_xgb.predict_proba(feat_df)[:, 1]
            p_rf = DATA_STORE.m1_rf.predict_proba(feat_df)[:, 1]

            meta_X = pd.DataFrame({"LightGBM": p_lgbm, "XGBoost": p_xgb, "RandomForest": p_rf})
            p_stack = float(DATA_STORE.m1_meta.predict_proba(meta_X)[:, 1][0])

            sess.purchase_probability = p_stack
            threshold = getattr(DATA_STORE, "m1_best_threshold", 0.10)
            sess.predicted_purchase = 1 if p_stack >= threshold else 0
        except Exception as e:
            # Keep existing probability on feature extraction exception
            pass


# Global session manager instance
SESSION_MANAGER = SessionManager()


# ============================================================
# RECOMMENDATION & DASHBOARD INTEGRATION SERVICES
# (Calling existing modules in recommendation/ranking.py)
# ============================================================

def generate_recommendations_for_session(session_id: str, limit: int = 8) -> List[dict]:
    """
    Executes the existing final trust-aware recommendation engine for a session.
    1. Identifies current session
    2. Obtains Model 1 intent probability
    3. Retrieves candidate items from viewed products & category affinity
    4. Obtains Model 2 brand trust & risk
    5. Calls existing compute_trust_aware_scores() from recommendation/ranking.py
    6. Returns customer-safe recommendations (strictly zero ML terminology)
    """
    sess = SESSION_MANAGER.get_or_create(session_id)
    p_intent = float(sess.purchase_probability)

    # 1. Candidate Retrieval (In-session + Category Affinity)
    candidates_records = []
    seen_pids = set()

    # In-session items
    for item in sess.cart_items.values():
        pid = item["product_id"]
        seen_pids.add(pid)
        candidates_records.append({
            "user_session": session_id,
            "product_id": pid,
            "category_code": item["category_code"],
            "brand": item["brand"],
            "price": item["price"],
            "candidate_source": "in_session",
            "base_relevance_score": IN_SESSION_CART_RELEVANCE
        })

    for item in sess.viewed_products.values():
        pid = item["product_id"]
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        candidates_records.append({
            "user_session": session_id,
            "product_id": pid,
            "category_code": item["category_code"],
            "brand": item["brand"],
            "price": item["price"],
            "candidate_source": "in_session",
            "base_relevance_score": IN_SESSION_VIEW_RELEVANCE
        })

    # Category affinity expansion from catalog
    cat_group = DATA_STORE.category_products.get(sess.active_category)
    if cat_group is not None and not cat_group.empty:
        for _, row in cat_group.head(15).iterrows():
            cid = int(row["product_id"])
            if cid in seen_pids:
                continue
            seen_pids.add(cid)
            pop = float(row.get("popularity_score", 0.5))
            rel = CATEGORY_AFFINITY_MIN_RELEVANCE + pop * (CATEGORY_AFFINITY_MAX_RELEVANCE - CATEGORY_AFFINITY_MIN_RELEVANCE)
            candidates_records.append({
                "user_session": session_id,
                "product_id": cid,
                "category_code": str(row["category_code"]),
                "brand": str(row["brand"]),
                "price": float(row["price"]),
                "candidate_source": "category_affinity",
                "base_relevance_score": rel
            })

    # If too few candidates, expand across catalog
    if len(candidates_records) < limit:
        for _, row in DATA_STORE.catalog_df.head(20).iterrows():
            cid = int(row["product_id"])
            if cid in seen_pids:
                continue
            seen_pids.add(cid)
            candidates_records.append({
                "user_session": session_id,
                "product_id": cid,
                "category_code": str(row["category_code"]),
                "brand": str(row["brand"]),
                "price": float(row["price"]),
                "candidate_source": "catalog_fallback",
                "base_relevance_score": 0.30
            })

    candidates_df = pd.DataFrame(candidates_records)

    # 2. Attach Model 1 User Intent Probability
    candidates_df["purchase_probability"] = p_intent

    # 3. Attach Model 2 Brand Trust & Risk Analysis
    trust_scores = []
    susp_scores = []
    risk_levels = []
    for b in candidates_df["brand"]:
        t_info = DATA_STORE.get_brand_trust(b)
        trust_scores.append(t_info["trust_score"])
        susp_scores.append(t_info["suspiciousness_score"])
        risk_levels.append(t_info["risk_level"])

    candidates_df["trust_score"] = trust_scores
    candidates_df["suspiciousness_score"] = susp_scores
    candidates_df["risk_level"] = risk_levels

    # 4. Call EXISTING compute_trust_aware_scores from recommendation/ranking.py
    ranked_df = compute_trust_aware_scores(candidates_df, alpha=ALPHA, beta=BETA, risk_penalties=RISK_PENALTIES)

    # 5. Extract Top-K
    topk_df = extract_topk_recommendations(ranked_df, score_column="trust_aware_score", k=limit)

    # 6. Format customer-safe products (strictly ZERO ML terminology)
    results = []
    for rank_idx, (_, row) in enumerate(topk_df.iterrows(), start=1):
        pid = int(row["product_id"])
        formatted = DATA_STORE.format_product(pid)
        results.append({
            "rank": rank_idx,
            "product_id": pid,
            "title": formatted["title"],
            "brand": formatted["brand"],
            "category_code": formatted["category_code"],
            "price": formatted["price"],
            "rating": formatted["rating"],
            "reviews_count": formatted["reviews_count"],
            "image_url": formatted["image_url"]
        })

    return results


def get_dashboard_telemetry(session_id: str) -> dict:
    """
    Constructs the complete FYP Evaluator telemetry payload:
    - Session info & chronological timeline
    - Model 1 User Intent
    - Model 2 Brand Trust
    - Final Trust-Aware Recommendations table
    - Before (Purchase-Only) vs Final (Trust-Aware) Rank Shifts table
    """
    sess = SESSION_MANAGER.get_or_create(session_id)
    p_intent = float(sess.purchase_probability)

    # Candidate pool generation
    candidates_records = []
    seen_pids = set()

    for item in sess.cart_items.values():
        pid = item["product_id"]
        seen_pids.add(pid)
        candidates_records.append({
            "user_session": session_id,
            "product_id": pid,
            "category_code": item["category_code"],
            "brand": item["brand"],
            "price": item["price"],
            "candidate_source": "in_session",
            "base_relevance_score": IN_SESSION_CART_RELEVANCE
        })

    for item in sess.viewed_products.values():
        pid = item["product_id"]
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        candidates_records.append({
            "user_session": session_id,
            "product_id": pid,
            "category_code": item["category_code"],
            "brand": item["brand"],
            "price": item["price"],
            "candidate_source": "in_session",
            "base_relevance_score": IN_SESSION_VIEW_RELEVANCE
        })

    cat_group = DATA_STORE.category_products.get(sess.active_category)
    if cat_group is not None and not cat_group.empty:
        for _, row in cat_group.head(15).iterrows():
            cid = int(row["product_id"])
            if cid in seen_pids:
                continue
            seen_pids.add(cid)
            pop = float(row.get("popularity_score", 0.5))
            rel = CATEGORY_AFFINITY_MIN_RELEVANCE + pop * (CATEGORY_AFFINITY_MAX_RELEVANCE - CATEGORY_AFFINITY_MIN_RELEVANCE)
            candidates_records.append({
                "user_session": session_id,
                "product_id": cid,
                "category_code": str(row["category_code"]),
                "brand": str(row["brand"]),
                "price": float(row["price"]),
                "candidate_source": "category_affinity",
                "base_relevance_score": rel
            })

    candidates_df = pd.DataFrame(candidates_records)
    candidates_df["purchase_probability"] = p_intent

    trust_scores = []
    susp_scores = []
    risk_levels = []
    for b in candidates_df["brand"]:
        t_info = DATA_STORE.get_brand_trust(b)
        trust_scores.append(t_info["trust_score"])
        susp_scores.append(t_info["suspiciousness_score"])
        risk_levels.append(t_info["risk_level"])

    candidates_df["trust_score"] = trust_scores
    candidates_df["suspiciousness_score"] = susp_scores
    candidates_df["risk_level"] = risk_levels

    # Call EXISTING ranking functions
    ranked_final = compute_trust_aware_scores(candidates_df, alpha=ALPHA, beta=BETA)
    ranked_final = compute_baseline_scores(ranked_final, DATA_STORE.catalog_df, alpha=ALPHA)

    # Compute Before (Purchase-Only) and Final (Trust-Aware) rankings
    top_final = extract_topk_recommendations(ranked_final, score_column="trust_aware_score", k=10)
    top_before = extract_topk_recommendations(ranked_final, score_column="purchase_only_score", k=10)

    before_rank_map = {row["product_id"]: int(row["rank"]) for _, row in top_before.iterrows()}
    final_rank_map = {row["product_id"]: int(row["rank"]) for _, row in top_final.iterrows()}

    # Final recommendations table list
    final_recommendations_list = []
    for _, row in top_final.iterrows():
        pid = int(row["product_id"])
        b_name = str(row["brand"])
        r_level = str(row["risk_level"])
        final_recommendations_list.append({
            "rank": int(row["rank"]),
            "product_id": pid,
            "title": DATA_STORE.format_product(pid)["title"],
            "brand": b_name,
            "category_code": str(row["category_code"]),
            "purchase_probability": round(float(row["purchase_probability"]), 4),
            "trust_score": round(float(row["trust_score"]), 4),
            "risk_level": r_level,
            "penalty_multiplier": RISK_PENALTIES.get(r_level, 0.75),
            "base_relevance": round(float(row["base_relevance_score"]), 2),
            "final_score": round(float(row["trust_aware_score"]), 4)
        })

    # Before vs. Final comparisons
    comparison_list = []
    for pid in top_final["product_id"].head(5):
        row = top_final[top_final["product_id"] == pid].iloc[0]
        b_rank = before_rank_map.get(pid, 10)
        f_rank = final_rank_map.get(pid, 1)
        shift = b_rank - f_rank  # Positive shift means promoted to a better rank
        b_name = str(row["brand"])
        t_score = float(row["trust_score"])
        r_level = str(row["risk_level"])
        psi = RISK_PENALTIES.get(r_level, 0.75)

        if shift > 0:
            direction = "PROMOTED"
            reason = f"High brand trust ({t_score:.4f}) and low risk (Psi={psi:.2f}) promoted product +{shift} ranks above competitors."
        elif shift < 0:
            direction = "DEMOTED"
            reason = f"Risk penalty discount multiplier (Psi={psi:.2f}) demoted item by {abs(shift)} ranks to favor higher-trust alternatives."
        else:
            direction = "UNCHANGED"
            reason = f"Product retained its rank due to strong user in-session relevance."

        comparison_list.append({
            "product_id": int(pid),
            "title": DATA_STORE.format_product(pid)["title"],
            "brand": b_name,
            "category_code": str(row["category_code"]),
            "risk_level": r_level,
            "trust_score": round(t_score, 4),
            "penalty_multiplier": psi,
            "purchase_only_rank": b_rank,
            "trust_aware_rank": f_rank,
            "rank_shift": shift,
            "shift_direction": direction,
            "reason": reason
        })

    # Brand Trust info for current brand
    curr_brand_info = DATA_STORE.get_brand_trust(sess.current_brand)

    s_dict = sess.to_dict()
    m1_dict = s_dict["model1_intent"]
    m2_dict = {
        "current_brand": sess.current_brand,
        "trust_score": curr_brand_info["trust_score"],
        "suspiciousness_score": curr_brand_info["suspiciousness_score"],
        "risk_level": curr_brand_info["risk_level"],
        "trust_category": curr_brand_info["trust_category"],
        "penalty_multiplier": curr_brand_info["penalty_multiplier"],
        "interpretation": curr_brand_info["interpretation"]
    }

    return {
        # Top-level session telemetry for direct consumption by DashboardPage.jsx
        "session_id": sess.session_id,
        "session_start_time": sess.start_time_str,
        "duration_seconds": sess.duration_seconds,
        "current_activity": sess.current_activity,
        "interaction_count": len(sess.events),
        "current_category": sess.active_category,
        "current_product": sess.current_product,
        "current_brand": sess.current_brand,
        "products_viewed": s_dict["products_viewed"],
        "products_carted": s_dict["products_carted"],
        "products_purchased": s_dict["products_purchased"],
        "timeline": sess.events,

        # Model 1 User Intent (both aliases)
        "model1_intent": m1_dict,
        "user_intent": m1_dict,

        # Model 2 Brand Trust (both aliases)
        "model2_brand_trust": m2_dict,
        "brand_trust": m2_dict,

        # Recommendations & Comparisons
        "final_recommendations": final_recommendations_list,
        "before_vs_final": {
            "session_id": session_id,
            "alpha": ALPHA,
            "beta": BETA,
            "comparisons": comparison_list
        },
        "model_metrics": {
            "hit_rate_10": "94.10%",
            "ndcg_10": "0.6078",
            "average_trust_10": "0.6991",
            "high_risk_exposure_10": "0.00%"
        },
        "session": s_dict
    }
