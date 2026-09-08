# SYSTEM ARCHITECTURE
## Trust-Aware E-Commerce Recommendation Using Brand Risk Analysis

**Project Title:** Trust-Aware E-Commerce Recommendation Using Brand Risk Analysis  
**Goal:** Deliver a working end-to-end prototype (Customer Store + FYP Evaluation Dashboard + FastAPI Backend + Pretrained ML Models) optimized for speed, simplicity, and zero retrain/redesign of existing ML models.

---

## 1. High-Level Architecture Overview

The system runs entirely on the local machine with two primary layers:
1. **Frontend (React + Vite):** A unified single-page application hosting both **Website 1 (Customer Store)** and **Website 2 (Admin / FYP Evaluation Dashboard)**.
2. **Backend (FastAPI):** A high-performance Python web server that directly imports existing recommendation modules, loads serialized Model 1 & Model 2 artifacts into memory, and tracks active session clickstreams.

```
+-------------------------------------------------------------------------------+
|                             FRONTEND (React + Vite)                           |
|                                                                               |
|   [ Website 1: Customer Store ]         [ Website 2: Admin / FYP Dashboard ]   |
|   - URL: /                              - URL: /admin                         |
|   - Clean commercial storefront         - Live session telemetry timeline     |
|   - Single 'Recommended for You' block  - Model 1 purchase probability meter  |
|   - Zero ML jargon / internal scores    - Model 2 brand trust & risk badges   |
|                                         - Before vs. Final rank shift table   |
+-------------------------------------------------------------------------------+
                                  |                     |
                      User Events | Recommendations     | Diagnostic Polling
                      (POST/GET)  | (JSON)              | (GET)
                                  v                     v
+-------------------------------------------------------------------------------+
|                            BACKEND (FastAPI Server)                           |
|                                                                               |
|   - In-Memory Session Store (tracks events, cart items, active category)      |
|   - Catalog & Brand Cache (O(1) in-memory lookups for 30,224 products)        |
|                                                                               |
|   +--------------------------+  +--------------------------+                  |
|   |  Model 1 Inference       |  |  Model 2 Trust Lookup    |                  |
|   |  - LGBM + XGB + RF       |  |  - 766 Scored Brands     |                  |
|   |  - Stacking Meta-Learner |  |  - Standardized Fallback |                  |
|   +--------------------------+  +--------------------------+                  |
|                                  |                                            |
|                                  v                                            |
|   +-----------------------------------------------------------------------+   |
|   |  Final Recommendation Engine (recommendation/ranking.py)              |   |
|   |  FinalScore = R(s,p) * (1 + alpha * P(purchase)) * (Trust^beta) * Psi |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

---

## 2. Technology Stack

| Layer | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend** | **React 18 + Vite** | Instant HMR, minimal build overhead, component-based modularity. |
| **Styling** | **Vanilla CSS** | CSS variables, modern dark/light themes, zero external CSS framework fragility. |
| **Icons** | **Lucide-React** | Clean, lightweight SVG iconography. |
| **Backend** | **FastAPI + Uvicorn** | Native Python async server, zero friction importing existing ML scripts, sub-5ms local responses. |
| **Data Engine** | **Pandas + In-Memory Dictionaries** | Instant lookups for preprocessed datasets (`product_catalog.csv`, `final_brand_trust_dataset.csv`). |
| **Model 1 Engine** | **Joblib + Scikit-Learn + LightGBM + XGBoost** | Loads existing `.pkl` stacking ensemble models directly without modifications. |
| **Model 2 Engine** | **Direct CSV Lookup + Fallback Policy** | Pre-computed brand risk classifications (`Low`, `Medium`, `High`) and trust scores $\in [0.0, 1.0]$. |
| **Infrastructure** | **Zero Docker / Zero External DB / Zero Redis** | Completely self-contained Python process and static Vite dev server. |

---

## 3. Session Management & Synchronization

To synchronize **Website 1 (Customer Store)** and **Website 2 (Admin Dashboard)** seamlessly without a database:
1. **Shared Session UUID:**
   - Website 1 generates or retrieves `session_id` (stored in `localStorage` or URL query parameter `?session_id=...`).
   - Website 2 reads the active `session_id` or allows switching between the live user session and verified benchmark sessions (`0f65dee0-...`, `2bb8e316-...`, `5a5350f2-...`).
2. **In-Memory Backend Session Store:**
   - The FastAPI backend stores active sessions in a Python dictionary:
     ```python
     sessions: Dict[str, dict] = {
         "session_id": {
             "created_at": "2026-09-08T14:30:00Z",
             "events": [{"event_type": "view", "product_id": 1500227, "timestamp": "..."}, ...],
             "cart": [{"product_id": 1500227, "quantity": 1}],
             "last_category": "computers.peripherals.printer",
             "purchase_probability": 0.8105,
             "predicted_purchase": 1,
             "recommendations": [...]
         }
     }
     ```
3. **Live Sync:**
   - Website 2 polls `/api/admin/session-state?session_id=...` periodically (e.g. every 1.5 seconds) to automatically update the timeline, Model 1 intent meter, and recommendation shifts as the customer clicks on Website 1.

---

## 4. API Endpoints Specification

### 4.1. Catalog & Product Endpoints

#### `GET /api/catalog`
* **Purpose:** Browse and search products on Website 1.
* **Query Parameters:**
  * `page` (int, default: 1)
  * `limit` (int, default: 12)
  * `category` (str, optional)
  * `brand` (str, optional)
  * `search` (str, optional)
* **Response (200 OK):**
```json
{
  "total": 30224,
  "page": 1,
  "limit": 12,
  "products": [
    {
      "product_id": 1500227,
      "title": "Epson Printer - Model #1500227",
      "category_code": "computers.peripherals.printer",
      "brand": "epson",
      "price": 149.99,
      "rating": 4.7,
      "reviews_count": 128,
      "image_url": "/images/categories/printer.jpg",
      "popularity_score": 0.84
    }
  ]
}
```

#### `GET /api/products/{product_id}`
* **Purpose:** View details for a specific product.
* **Path Parameter:** `product_id` (int)
* **Response (200 OK):**
```json
{
  "product_id": 1500227,
  "title": "Epson Printer - Model #1500227",
  "category_code": "computers.peripherals.printer",
  "brand": "epson",
  "price": 149.99,
  "rating": 4.7,
  "reviews_count": 128,
  "description": "High-efficiency commercial printer with eco-tank integration.",
  "specs": {
    "Brand": "Epson",
    "Category": "Printers & Peripherals",
    "Model Number": "1500227",
    "Connectivity": "Wireless / USB"
  }
}
```

---

### 4.2. Event Tracking & Session Endpoints

#### `POST /api/events`
* **Purpose:** Log a customer interaction from Website 1 (`view`, `cart`, `purchase`, `session_start`).
* **Request Body:**
```json
{
  "session_id": "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
  "event_type": "view",
  "product_id": 1500227
}
```
* **Response (200 OK):**
```json
{
  "status": "success",
  "session_id": "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
  "event_count": 4,
  "cart_count": 1,
  "purchase_probability": 0.8105,
  "predicted_purchase": 1
}
```

---

### 4.3. Recommendation Endpoints

#### `GET /api/recommendations`
* **Purpose:** Returns customer-safe Top-K recommendations for Website 1 (strictly zero ML terminology).
* **Query Parameters:**
  * `session_id` (str, required)
  * `limit` (int, default: 8)
* **Response (200 OK):**
```json
{
  "session_id": "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
  "recommendations": [
    {
      "rank": 1,
      "product_id": 1500208,
      "title": "Xerox Printer - Model #1500208",
      "brand": "xerox",
      "category_code": "computers.peripherals.printer",
      "price": 139.99,
      "rating": 4.8,
      "image_url": "/images/categories/printer.jpg"
    },
    {
      "rank": 2,
      "product_id": 1500075,
      "title": "Canon Printer - Model #1500075",
      "brand": "canon",
      "category_code": "computers.peripherals.printer",
      "price": 129.99,
      "rating": 4.6,
      "image_url": "/images/categories/printer.jpg"
    }
  ]
}
```

---

### 4.4. Admin / FYP Evaluation Dashboard Endpoints

#### `GET /api/admin/session-state`
* **Purpose:** Provides comprehensive real-time telemetry for Website 2.
* **Query Parameters:**
  * `session_id` (str, required)
* **Response (200 OK):**
```json
{
  "session_id": "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
  "duration_seconds": 252,
  "active_category": "computers.peripherals.printer",
  "events_count": 6,
  "cart_items": [
    {
      "product_id": 1500227,
      "title": "Epson Printer - Model #1500227",
      "brand": "epson",
      "price": 149.99,
      "quantity": 1
    }
  ],
  "timeline": [
    { "timestamp": "14:02:10", "event_type": "session_start", "detail": "Session initialized" },
    { "timestamp": "14:02:25", "event_type": "view", "product_id": 1500227, "brand": "epson" },
    { "timestamp": "14:03:10", "event_type": "cart", "product_id": 1500227, "brand": "epson" }
  ],
  "model1_intent": {
    "purchase_probability": 0.8105,
    "predicted_purchase": 1,
    "intent_level": "High Purchase Intent",
    "intent_multiplier": 1.8105,
    "model_name": "Stacking Meta-Ensemble (LightGBM + XGBoost + RF)"
  }
}
```

#### `GET /api/admin/rank-comparison`
* **Purpose:** Powers the Before (Purchase-Only) vs. Final (Trust-Aware) rank shift table on Website 2.
* **Query Parameters:**
  * `session_id` (str, required)
* **Response (200 OK):**
```json
{
  "session_id": "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
  "alpha": 1.0,
  "beta": 1.0,
  "comparisons": [
    {
      "product_id": 1500208,
      "brand": "xerox",
      "category_code": "computers.peripherals.printer",
      "trust_score": 0.9107,
      "risk_level": "Low",
      "penalty_multiplier": 1.0,
      "purchase_only_rank": 11,
      "trust_aware_rank": 3,
      "rank_shift": 8,
      "shift_direction": "PROMOTED",
      "reason": "High trust score (0.9107) and zero risk penalty boosted product above competitors."
    },
    {
      "product_id": 1500447,
      "brand": "hp",
      "category_code": "computers.peripherals.printer",
      "trust_score": 0.6679,
      "risk_level": "Medium",
      "penalty_multiplier": 0.75,
      "purchase_only_rank": 3,
      "trust_aware_rank": 8,
      "rank_shift": -5,
      "shift_direction": "DEMOTED",
      "reason": "Medium risk penalty (Psi = 0.75) suppressed ranking to favor higher-trust alternatives."
    }
  ]
}
```

#### `GET /api/admin/metrics`
* **Purpose:** Returns verified academic evaluation metrics from `recommendation/data/model_comparison_table.csv`.
* **Response (200 OK):**
```json
{
  "models": [
    {
      "name": "Trust-Aware (Proposed)",
      "hit_rate_10": "94.10%",
      "ndcg_10": "0.6078",
      "average_trust_10": "0.6991",
      "high_risk_exposure_10": "0.00%"
    },
    {
      "name": "Purchase-Only Baseline",
      "hit_rate_10": "92.10%",
      "ndcg_10": "0.5349",
      "average_trust_10": "0.5930",
      "high_risk_exposure_10": "7.18%"
    },
    {
      "name": "Hard-Filter Safety Baseline",
      "hit_rate_10": "90.60%",
      "ndcg_10": "0.5269",
      "average_trust_10": "0.6320",
      "high_risk_exposure_10": "0.00%"
    }
  ]
}
```

#### `GET /api/admin/sample-sessions`
* **Purpose:** Quick session switcher presets for the thesis demonstration.
* **Response (200 OK):**
```json
[
  {
    "session_id": "0f65dee0-ae4d-460e-bb66-3da1bbbaec6b",
    "title": "Printer Shopper (High Intent, High Trust Promotions)",
    "category": "computers.peripherals.printer",
    "default_intent": 0.8105
  },
  {
    "session_id": "2bb8e316-9856-441e-a858-5723f916b456",
    "title": "Kids Toys (Significant Unknown Brand Demotions)",
    "category": "kids.toys",
    "default_intent": 0.1200
  },
  {
    "session_id": "5a5350f2-c5d7-4230-8fec-3ac32c134ab3",
    "title": "Smartphone Search (Xiaomi vs Samsung Re-ordering)",
    "category": "electronics.smartphone",
    "default_intent": 0.4500
  }
]
```

---

## 5. End-to-End Data & Model Execution Flows

### 5.1. Website 1 → Backend Flow
1. Customer performs an action (e.g. loads home page, views product #1500227, adds product to cart).
2. Website 1 dispatches `POST /api/events` with `{ session_id, event_type, product_id }`.
3. Backend appends event to the active session list and recalculates session state.
4. Website 1 requests recommendations via `GET /api/recommendations?session_id=...`.
5. Backend returns customer-safe product cards (title, image, price, rating) without leaking trust scores.

---

### 5.2. Backend → Model 1 (User Intent) Flow
1. When events arrive for a session, the backend extracts the 18 feature attributes required by Model 1:
   - `event_count`, `view_count`, `cart_count`, `unique_products`, `unique_categories`, `unique_brands`
   - `average_price`, `max_price`, `min_price`, `session_duration`, `hour`, `day_of_week`, `is_weekend`
   - `last_action_1` to `last_action_5` (mapped: 1 for view, 2 for cart)
2. Backend checks if the session is one of the 1,000 pre-evaluated sessions in `user-intent/data/purchase_prediction_test_results.csv`:
   - If found: returns verified probability $P(\text{purchase} \mid s)$.
   - If custom live session: passes features into serialized `purchase_lgbm.pkl`, `purchase_xgb.pkl`, `purchase_rf.pkl`, then feeds base probabilities into `purchase_stacking_meta.pkl`.
   - Fallback: empirical base purchase rate ($0.05$).
3. Output: $P(\text{purchase} \mid s) \in [0.0, 1.0]$ and $\text{Intent Multiplier} = 1.0 + \alpha \cdot P$.

---

### 5.3. Backend → Model 2 (Brand Trust & Risk) Flow
1. For every candidate product, the backend retrieves the brand name (lowercase, stripped).
2. Backend queries the in-memory lookup table loaded from `behavioral-analysis/data/processed/final_brand_trust_dataset.csv`:
   - **Case A: Brand is known and scored (766 brands):**
     - Retrieves `trust_score` $\in [0.1500, 0.9995]$.
     - Retrieves `risk_level` (`Low`, `Medium`, `High`).
     - Lookups penalty multiplier $\Psi(b)$: Low $\to 1.00$, Medium $\to 0.75$, High $\to 0.20$.
     - Flag: `scoring_status = "scored"`.
   - **Case B: Unbranded / generic (`brand == "unknown"`):**
     - Fallback: `trust_score = 0.50`, `risk_level = "Medium"`, $\Psi = 0.75$.
     - Flag: `scoring_status = "unbranded_fallback"`.
   - **Case C: Unscored brand (<5 historical events):**
     - Fallback: `trust_score = 0.75`, `risk_level = "Medium"`, $\Psi = 0.75$.
     - Flag: `scoring_status = "insufficient_evidence_fallback"`.

---

### 5.4. Backend → Final Recommendation Flow
1. **Candidate Retrieval:**
   - In-session items: Viewed or carted items in the current session (Base Relevance $R = 0.80$ for views, $1.00$ for cart).
   - Category affinity items: Top items from the current active category in `product_catalog.csv` (Base Relevance $R \in [0.20, 0.60]$).
2. **Scoring via Existing Module (`recommendation/ranking.py`):**
   - Calls existing `compute_trust_aware_scores(candidates_df, alpha=1.0, beta=1.0)`:
     $$\text{FinalScore}(s, p) = R(s, p) \times (1.0 + 1.0 \times P) \times (\text{Trust}(b))^{1.0} \times \Psi(b)$$
   - Calls existing `compute_baseline_scores(candidates_df, catalog_df, alpha=1.0)` to compute the baseline purchase-only score:
     $$\text{PurchaseOnlyScore}(s, p) = R(s, p) \times (1.0 + 1.0 \times P)$$
3. **Ranking & Displacement Calculation:**
   - Calls `extract_topk_recommendations()` for both score columns.
   - Calculates rank shift: $\Delta = \text{Rank}_{\text{PurchaseOnly}} - \text{Rank}_{\text{TrustAware}}$.
   - Identifies whether the item was promoted ($\Delta > 0$), demoted ($\Delta < 0$), or unchanged ($\Delta = 0$).

---

### 5.5. Backend → Website 2 Dashboard Flow
1. Website 2 sends `GET /api/admin/session-state?session_id=...` and `GET /api/admin/rank-comparison?session_id=...`.
2. Backend returns:
   - Chronological event timeline (`view`, `cart`, duration).
   - Live Model 1 Purchase Probability gauge and Intent multiplier.
   - Model 2 Brand Trust diagnostic cards for all evaluated brands.
   - Live Before vs. Final rank comparison table showing rank displacements and the exact mathematical rationale.
   - Academic benchmark metrics (HitRate@10: 94.1%, NDCG@10: 0.6078, HighRiskExposure: 0.00%).

---

## 6. Directory Structure for Implementation

To finish the frontend and backend in minimal time, files are organized directly:

```
d:\FYP\implementation\
├── backend/                        # [NEW] Lightweight FastAPI Backend
│   ├── __init__.py
│   ├── server.py                   # Main FastAPI server & REST routes
│   └── service.py                  # Integration layer calling existing ML modules
│
├── frontend/                       # [NEW] Single React + Vite Application
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                 # View Switcher (Customer Store vs Admin Dashboard)
│       ├── index.css               # Clean modern CSS design system
│       ├── components/
│       │   ├── Navbar.jsx          # Header with Cart count & Switcher link
│       │   ├── ProductCard.jsx     # Commercial product card
│       │   ├── Recommendations.jsx # 'Recommended for You' carousel/shelf
│       │   └── CartDrawer.jsx      # Slide-out cart modal
│       ├── pages/
│       │   ├── StorePage.jsx       # Website 1: Customer Store
│       │   └── DashboardPage.jsx   # Website 2: FYP Evaluation Dashboard
│       └── services/
│           └── api.js              # Fetch client wrapper
│
├── recommendation/                 # [EXISTING - UNMODIFIED]
│   ├── ranking.py                  # Used directly by backend
│   ├── candidate_generator.py      # Used directly by backend
│   ├── trust_integrator.py         # Used directly by backend
│   └── data/                       # product_catalog.csv, candidates.csv, etc.
│
├── user-intent/                    # [EXISTING - UNMODIFIED]
│   ├── models/                     # purchase_lgbm.pkl, purchase_xgb.pkl, etc.
│   └── data/                       # test results & session probabilities
│
└── behavioral-analysis/            # [EXISTING - UNMODIFIED]
    ├── models/                     # isolation_forest.pkl, scaler.pkl
    └── data/processed/             # final_brand_trust_dataset.csv
```

---

## 7. Execution & Running Plan

1. **Backend:**
   ```bash
   python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Frontend:**
   ```bash
   cd frontend && npm run dev
   ```
3. **URL Endpoints for Evaluation:**
   * **Website 1 (Customer Store):** `http://localhost:5173/`
   * **Website 2 (Admin Dashboard):** `http://localhost:5173/#/admin` (or `/admin`)
   * **Side-by-Side Dual View:** Can be toggled with a single click in the navbar.
