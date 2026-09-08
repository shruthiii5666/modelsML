# RECOMMENDATION MODULE IMPLEMENTATION PLAN
## Trust-Aware E-Commerce Recommendation Using Brand Risk Analysis

**Final Year Project (FYP)**  
**Document Version:** 1.0  
**Status:** PROPOSED — PENDING USER APPROVAL (PHASE A DELIVERABLE)  
**Implementation Phase:** STRICTLY HALTED UNTIL USER CONFIRMATION  

---

## 1. Objective

The primary objective of the **Trust-Aware Recommendation Module** is to bridge the two completed machine learning models:
1. **User Purchase Intent Model** (supervised Stacking Ensemble predicting session-level purchase probability)
2. **Brand Risk & Trust Model** (unsupervised Isolation Forest predicting category-relative behavioral anomalies and brand trustworthiness)

### Conceptual Problem
Conventional e-commerce recommender systems prioritize purely conversion likelihood or engagement signals. Consequently, if a user exhibits high purchase intent while viewing products from anomalous, deceitful, or high-risk brands (e.g., brands with artificial conversion spikes, abnormal cart abandonment patterns, or erratic price fluctuations), standard models will continue pushing those hazardous items to maximize immediate click/purchase metrics.

### System Goal
This module develops a **Trust-Aware E-Commerce Recommendation System** that:
- Ingests active user session browsing behavior to determine conversion readiness ($P(\text{purchase} \mid \text{session})$).
- Retrieves and links product candidates to their manufacturing/distributing brand entities.
- Enriches candidate items with precomputed empirical Brand Trust Scores ($\text{Trust}(b) \in [0, 1]$) and Risk Levels (`Low`, `Medium`, `High`).
- Applies a parameterized, explainable trust-aware ranking algorithm balancing candidate relevance, conversion probability, and brand trust.
- Demotes or filters high-risk, suspicious brands while promoting reliable, high-trust alternatives without destroying user session relevance.
- Demonstrates clear, measurable safety improvements over purchase-only and popularity baselines for the FYP viva and dissertation.

```text
       Clickstream Session Interactions
                      │
                      ▼
   ┌─────────────────────────────────────┐
   │        Candidate Generation         │  (In-session + Category Co-occurrence)
   └──────────────────┬──────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│ Model 1 Output   │      │ Model 2 Output   │
│ Purchase Intent  │      │ Brand Risk/Trust │
│ P(Purchase|S)    │      │ Trust(b), Risk   │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
   ┌─────────────────────────────────────┐
   │     Trust-Aware Ranking Engine      │
   │  S(s,p) = Relevance × P^α × Trust^β │
   └──────────────────┬──────────────────┘
                      ▼
   ┌─────────────────────────────────────┐
   │        Top-K Recommendations        │
   │      (Safe, High-Trust, Relevant)   │
   └─────────────────────────────────────┘
```

---

## 2. Existing Inputs (Completed Models)

Both models have been audited, locally verified, and executed. The recommendation module will consume their exact, finalized output artifacts without modification or retraining.

### A. Purchase Prediction Model (Model 1)
- **File Path:** `d:/FYP/implementation/user-intent/data/session_purchase_probabilities.csv`
- **Secondary Evaluation File:** `d:/FYP/implementation/user-intent/data/purchase_prediction_test_results.csv`
- **Model Checkpoints:** `user-intent/models/purchase_lgbm.pkl`, `purchase_xgb.pkl`, `purchase_rf.pkl`, `purchase_stacking_meta.pkl`
- **Exact Columns:**
  1. `user_session` (*str*): Unique UUID representing the e-commerce browsing session.
  2. `actual_purchase` (*int* $\in \{0, 1\}$): Ground-truth binary purchase indicator observed in the raw session.
  3. `purchase_probability` (*float64* $\in [0.0, 1.0]$): Continuous probability emitted by the Stacking Meta-Learner (Logistic Regression combining LightGBM, XGBoost, and Random Forest base predictions on 18 session features).
  4. `predicted_purchase` (*int* $\in \{0, 1\}$): Binary decision at the optimal F1 threshold ($0.10$).
- **How It Is Consumed:**
  For any candidate session $s$, `purchase_probability` serves as the scaling factor representing the user's conversion urgency. High purchase probabilities amplify relevance, while low purchase probabilities prioritize exploration.

### B. Brand Risk & Trust Model (Model 2)
- **Primary File Path:** `d:/FYP/implementation/behavioral-analysis/data/processed/final_brand_trust_dataset.csv`
- **Secondary File Path:** `d:/FYP/implementation/behavioral-analysis/data/output/brand_scores.csv`
- **Model Checkpoint:** `behavioral-analysis/models/isolation_forest.pkl`
- **Exact Columns:**
  1. `brand` (*str*): Lowercase normalized brand name (765 scored brands in current dataset).
  2. `analyzed_categories` (*int*): Count of distinct categories the brand was evaluated in.
  3. `anomalous_categories` (*int*): Number of categories where the brand was flagged as anomalous by the Isolation Forest.
  4. `anomaly_category_ratio` (*float* $\in [0.0, 1.0]$): $\frac{\text{anomalous\_categories}}{\text{analyzed\_categories}}$.
  5. `mean_anomaly_score` (*float*): Average normalized anomaly score across evaluated categories.
  6. `max_anomaly_score` (*float*): Peak normalized anomaly score across evaluated categories.
  7. `evidence_weighted_anomaly_score` (*float*): Category anomalies weighted by view/session volume.
  8. `suspiciousness_score` (*float* $\in [0.0, 1.0]$): Composite behavioral risk score computed in Phase 9 (higher means more suspicious).
  9. `risk_level` (*str*): Categorical band derived from suspiciousness:
     - `'Low'`: suspiciousness $< 0.33$ (710 brands)
     - `'Medium'`: $0.33 \le \text{suspiciousness} \le 0.67$ (49 brands)
     - `'High'`: suspiciousness $> 0.67$ (6 brands, e.g., `portcase`, `ariston`, `starline`, `oppo`, `janome`, `huggies`)
  10. `trust_score` (*float* $\in [0.0, 1.0]$): Core brand reliability score defined as $\text{trust\_score} = 1.0 - \text{suspiciousness\_score}$.
  11. `trust_category` (*str*): Qualitative trust tier:
      - `'High Trust'` ($\text{trust} > 0.67$)
      - `'Medium Trust'` ($0.33 < \text{trust} \le 0.67$)
      - `'Low Trust'` ($\text{trust} \le 0.33$)
      - `'Insufficient Evidence'` (unscored brands)
  12. `trust_interpretation` (*str*): Human-readable textual rationale generated in Phase 10.
  13. `scoring_status` (*str*): `'scored'` vs `'insufficient_evidence'`.
- **How It Is Consumed:**
  Every product candidate is joined on `brand`. Its `trust_score`, `risk_level`, and `trust_category` determine the trust discount/boost and risk penalty applied during ranking.

---

## 3. Data Flow Architecture

The data pipeline connects raw events to final Top-K recommendations through the following verified stages:

```text
[Stage 1: Raw Clickstream Catalog]
d:/FYP/implementation/2019-Oct.csv (events: view, cart)
                      │
                      ▼
[Stage 2: Session Extraction & Candidate Pool]
recommendation/candidate_generator.py
Extracts active test sessions + Candidate Products per session
                      │
                      ▼
[Stage 3: Product-to-Brand Relational Mapping]
Extracts unique (product_id, category_code, brand, price)
                      │
                      ▼
[Stage 4: Multi-Model Join & Fallback Policies]
recommendation/trust_integrator.py
Joins Model 1: purchase_probability (by user_session)
Joins Model 2: trust_score, risk_level (by brand)
Applies configurable defaults for missing/unscored/unknown brands
                      │
                      ▼
[Stage 5: Trust-Aware Scoring & Ranking]
recommendation/ranking.py
Computes FinalScore = Relevance × P(Purchase)^α × Trust^β × RiskPenalty
Sorts candidates descending per session
                      │
                      ▼
[Stage 6: Top-K Recommendation & Baseline Comparison]
recommendation/evaluation.py & run_recommendation.py
Generates Top-K recommendations for Proposed vs Baselines
Computes Utility (NDCG, HitRate) & Safety (AvgTrust, HighRiskExposure)
```

---

## 4. Candidate Generation Design

### Analysis of Available Strategies
1. **Strategy A (In-Session Only):** Re-ranking products already interacted with in session $s$.
   *Limitation:* Many sessions have only 3–5 interactions. If $K=10$, in-session items alone cannot fill the Top-K list.
2. **Strategy B (Full Matrix Factorization / Global CF):** Fitting SVD/ALS across 42 million interactions.
   *Limitation:* Prohibitive RAM and compute requirements on a local Windows machine.
3. **Strategy C (Category-Affinity & Co-occurrence Hybrid) [RECOMMENDED]:**
   - For an active session $s$, retrieve all items viewed or carted in $s$ (**In-Session Candidates**).
   - Expand the candidate pool with top co-viewed and popular items from the same categories browsed in $s$ (**Category-Affinity Candidates**).
   - Generates a rich pool of 20 to 50 relevant candidate products per session.

### Computational Efficiency
- The candidate generator extracts an offline **Item-Category-Brand Catalog** and a **Category Item Co-occurrence / Popularity Index** using the verified 100,000-row test split (or 3-chunk dataset).
- Generation runs in milliseconds per session in memory without touching the 5.6GB raw file during recommendation inference.

---

## 5. Integration Dataframe Specification

The central dataframe connecting all components is named **`recommendation_candidates_df`**.

| Column Name | Datatype | Source | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `user_session` | `object` (str) | Model 1 & Clickstream | Active session identifier |
| `product_id` | `int64` | Clickstream Catalog | Candidate product ID |
| `category_code` | `object` (str) | Clickstream Catalog | Category classification (e.g. `electronics.smartphone`) |
| `brand` | `object` (str) | Clickstream Catalog | Lowercase normalized brand name |
| `price` | `float64` | Clickstream Catalog | Product price |
| `candidate_source` | `object` (str) | Candidate Generator | `'in_session'` vs `'category_affinity'` |
| `base_relevance_score` | `float64` | Candidate Generator | Normalized relevance $[0.1, 1.0]$ based on session interaction & category affinity |
| `purchase_probability` | `float64` | Model 1 Output | Session purchase probability from Stacking Meta-Model |
| `predicted_purchase` | `int64` | Model 1 Output | Model 1 binary conversion flag ($0$ or $1$) |
| `trust_score` | `float64` | Model 2 Output | Brand trust score $\in [0.0, 1.0]$ (with fallback policy applied) |
| `suspiciousness_score` | `float64` | Model 2 Output | Anomaly score $\in [0.0, 1.0]$ from Phase 9 |
| `risk_level` | `object` (str) | Model 2 Output | `'Low'`, `'Medium'`, `'High'` |
| `trust_category` | `object` (str) | Model 2 Output | `'High Trust'`, `'Medium Trust'`, `'Low Trust'`, `'Insufficient Evidence'`, `'Unknown Brand'` |
| `final_score` | `float64` | Ranking Module | Computed trust-aware ranking score |
| `rank` | `int64` | Ranking Module | Integer position (1 to $K$) within the session |

---

## 6. Product-to-Brand Mapping & Fallback Policy

### Brand Name Normalization Rules
1. **String Cleaning:** All brand names are lowercased and stripped of leading/trailing whitespace:
   $$\text{brand} = \text{str}(\text{raw\_brand}).\text{strip}().\text{lower}()$$
2. **Missing Brand Values:** Any `NaN`, `None`, or empty string `""` is converted to `"unknown"`.
3. **Duplicate Mappings:** If a `product_id` appears with conflicting brands across sessions, the most frequent non-unknown brand is chosen as canonical.

---

## 7. Trust Score Assignment & Fallback Policies

Every candidate product must receive an explicit, justified trust score. Arbitrary silent imputation is strictly prohibited.

```text
Product Candidate
       │
       ├─► brand == 'unknown' ──────────────────────────► Trust = 0.50, Risk = 'Medium', Category = 'Unknown Brand'
       │
       ├─► brand in Model 2 Scored Dataset (765 brands)
       │         │
       │         ├─► Suspiciousness < 0.33 ─────────────► Exact Model Trust (> 0.67), Risk = 'Low', Category = 'High Trust'
       │         ├─► 0.33 <= Suspiciousness <= 0.67 ────► Exact Model Trust [0.33, 0.67], Risk = 'Medium', Category = 'Medium Trust'
       │         └─► Suspiciousness > 0.67 ─────────────► Exact Model Trust (< 0.33), Risk = 'High', Category = 'Low Trust'
       │
       └─► brand NOT in Model 2 Dataset (Insufficient Evidence)
                 │
                 └──────────────────────────────────────► Trust = 0.75, Risk = 'Low-Medium', Category = 'Insufficient Evidence'
```

### Configurable Fallback Policy Table

| Brand Condition | Fallback `trust_score` | Assigned `risk_level` | Assigned `trust_category` | Rationale & Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Scored Brands** (in Model 2) | Exact `trust_score` ($1 - \text{suspiciousness}$) | Exact `risk_level` (`Low`/`Medium`/`High`) | Exact `trust_category` | Direct empirical behavioral output from Phase 10. |
| **Insufficient Evidence Brands** | `DEFAULT_UNSCORED_TRUST = 0.75` | `'Medium'` | `'Insufficient Evidence'` | Conservative discount below the scored median ($0.8967$). Reflects unproven reliability without severe punishment. |
| **Unknown Brands** (`'unknown'`) | `DEFAULT_UNKNOWN_TRUST = 0.50` | `'Medium'` | `'Unknown Brand'` | Neutral midpoint penalty. Unbranded listings cannot guarantee quality or return compliance. |
| **Missing Risk Score** | `DEFAULT_UNSCORED_TRUST = 0.75` | `'Medium'` | `'Insufficient Evidence'` | Identical to unproven brand policy. |

---

## 8. Ranking Algorithm Design

### Mathematical Formula
For a given session $s$ and candidate product $p$ from brand $b$:

$$S(s, p) = R(s, p) \times \left( 1.0 + \alpha \cdot P(\text{purchase} \mid s) \right) \times \left( \text{Trust}(b) \right)^\beta \times \Psi(b)$$

Where:
- $R(s, p) \in [0.1, 1.0]$ is the **Base Relevance Score**:
  - $R(s, p) = 1.0$ for items added to cart in session $s$
  - $R(s, p) = 0.8$ for items viewed in session $s$
  - $R(s, p) \in [0.2, 0.6]$ for category-affinity items based on category popularity/co-occurrence
- $P(\text{purchase} \mid s) \in [0.0, 1.0]$ is the **Purchase Probability** from Model 1.
- $\alpha \ge 0$ is the **Purchase Intent Scaling Weight** (default $\alpha = 1.0$).
- $\text{Trust}(b) \in [0.0, 1.0]$ is the **Brand Trust Score** from Model 2.
- $\beta \ge 0$ is the **Brand Trust Sensitivity Exponent** (default $\beta = 1.0$).
- $\Psi(b)$ is the **Discrete Risk Demotion Multiplier**:
  $$\Psi(b) = \begin{cases} 
  1.00 & \text{if } \text{Risk Level is } \text{'Low'} \\
  0.75 & \text{if } \text{Risk Level is } \text{'Medium'} \\
  0.20 & \text{if } \text{Risk Level is } \text{'High'} \quad (\text{severe demotion})
  \end{cases}$$

### Why This Formula is Mathematically Sound for the FYP
1. **Continuous Trust Scaling ($\text{Trust}(b)^\beta$):** Smoothly rewards brands with trust $>0.90$ while penalizing brands with trust $<0.40$.
2. **Safety Net ($\Psi(b)$):** Guarantees that confirmed High-Risk anomaly brands (such as `portcase` with suspiciousness $0.8336$ and trust $0.1664$) receive a drastic penalty ($0.1664 \times 0.20 = 0.033$), preventing them from appearing in Top-K even if purchase intent is high.
3. **Controlled Baseline ($\beta = 0, \Psi = 1.0$):** Exactly recovers the **Purchase-Only Baseline**, allowing controlled A/B comparison.

---

## 9. Baselines for Comparative Evaluation

The module will implement and evaluate **four ranking models**:

1. **Baseline 1: Popularity-Only Ranking**
   - Ranks candidates purely by global category view/purchase popularity:
     $$S_{\text{pop}}(s, p) = \text{Popularity}(p)$$
2. **Baseline 2: Purchase-Only Intent Ranking (Current System without Trust)**
   - Ranks candidates using relevance and Model 1 purchase probability, ignoring brand risk:
     $$S_{\text{purchase}}(s, p) = R(s, p) \times \left( 1.0 + \alpha \cdot P(\text{purchase} \mid s) \right)$$
3. **Baseline 3: Hard-Filter Safety Model**
   - Ranks using Baseline 2, but completely purges/discards any product where $\text{Risk Level} == \text{'High'}$.
4. **Proposed System: Trust-Aware Continuous Ranking**
   - Full formula balancing relevance, purchase intent, trust exponent, and risk penalties.

---

## 10. Recommendation Output Specification

### A. Output Files Generated
1. `recommendation/data/topk_recommendations_trust_aware.csv`: Top-K recommendations per session under proposed model.
2. `recommendation/data/topk_recommendations_purchase_only.csv`: Top-K recommendations per session under Baseline 2.
3. `recommendation/data/recommendation_evaluation_summary.csv`: Metric table comparing all baselines across $K \in \{5, 10\}$.
4. `recommendation/data/before_after_demonstration_cases.csv`: Targeted case studies showing specific rank inversions for high-risk vs high-trust items.

### B. Exact Top-K Record Structure
Each row in the output recommendation file will contain:
- `user_session`: str
- `rank`: int ($1$ to $K$)
- `product_id`: int64
- `brand`: str
- `category_code`: str
- `price`: float
- `purchase_probability`: float (Model 1)
- `trust_score`: float (Model 2)
- `risk_level`: str (`Low`, `Medium`, `High`)
- `trust_category`: str
- `final_score`: float

---

## 11. Comprehensive Evaluation Framework

### A. Recommendation Utility Metrics (Relevance & Intent)
- **HitRate@K:** Proportion of sessions where at least one ground-truth purchased/carted product is in the Top-K.
- **Precision@K:** $\frac{\text{Number of relevant products in Top-K}}{K}$.
- **Recall@K:** $\frac{\text{Number of relevant products in Top-K}}{\text{Total relevant products in session}}$.
- **NDCG@K (Normalized Discounted Cumulative Gain):** Rewards placing relevant items at top rank positions (positions 1–3).

### B. Trust & Safety Metrics (The FYP Novelty)
- **Average Trust@K (ATS@K):**
  $$\text{ATS@K} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \frac{1}{K} \sum_{i=1}^K \text{Trust}(b_{s, i})$$
  *Goal:* Significant increase in ATS@K under the proposed system compared to purchase-only ranking.
- **High-Risk Exposure Rate (HRER@K):**
  $$\text{HRER@K} = \frac{\text{Total recommended items with Risk Level == 'High'}}{\text{Total items recommended across all sessions}}$$
  *Goal:* HRER@K reduced to $\approx 0\%$.
- **Low-Trust Demotion Rate:** Percentage of low-trust products displaced from Top-5 to rank $>10$.

---

## 12. Before vs. After Demonstration Design

To provide compelling evidence for the FYP defense, the module will generate side-by-side session comparisons:

### Illustrative Scenario
- **Session ID:** User viewing electronics / smartphone accessories.
- **Candidate Product A (Brand: `oppo`):** High-risk brand (Model 2 Suspiciousness = $0.7892$, Trust = $0.2108$).
- **Candidate Product B (Brand: `samsung` / `apple` / `globber`):** Low-risk brand (Model 2 Suspiciousness = $0.02$, Trust = $0.98$).

| Metric / Aspect | Purchase-Only Ranking (Baseline) | Trust-Aware Ranking (Proposed) | FYP Finding |
| :--- | :--- | :--- | :--- |
| **Product A (`oppo`, Low Trust)** | **Rank 1** (Score = $0.92$) | **Rank 8** (Score = $0.18$) | **Demoted by 7 positions** due to anomaly penalty. |
| **Product B (`samsung`, High Trust)** | **Rank 2** (Score = $0.88$) | **Rank 1** (Score = $0.86$) | **Promoted to Rank 1** due to superior brand trust. |
| **Session Safety** | High Risk Exposure | Safe, Trustworthy Recommendation | User protected from anomalous merchant behavior. |

---

## 13. Sensitivity Analysis Plan ($\beta$ Tuning)

A sweep across the Trust Sensitivity Parameter $\beta \in [0.0, 0.5, 1.0, 1.5, 2.0]$ will be evaluated:

| $\beta$ Value | Expected System Behavior | Measured Metrics |
| :--- | :--- | :--- |
| $\beta = 0.0$ | Trust disabled (Pure Purchase Intent Baseline) | Maximum utility, Highest risk exposure |
| $\beta = 0.5$ | Mild trust consideration | Moderate risk demotion, Minimal utility loss |
| $\beta = 1.0$ | **Default Proposed System** | Balanced utility & safety |
| $\beta = 1.5$ | Aggressive trust prioritization | Near-zero high-risk exposure, slight utility drop |
| $\beta = 2.0$ | Conservative safety mode | Zero high-risk exposure, High-Trust brands dominate |

---

## 14. Files to Create in the New Module

All new code will reside in a dedicated, clean package: `recommendation/`.

```text
d:/FYP/implementation/recommendation/
│
├── __init__.py
├── config.py                 # Central hyperparameter & path definitions
├── candidate_generator.py    # Generates in-session and category candidate items
├── trust_integrator.py       # Merges Model 1 + Model 2 outputs & applies fallbacks
├── ranking.py                # Computes final ranking formula & baseline scores
├── evaluation.py             # Computes HitRate@K, NDCG@K, ATS@K, HRER@K
├── run_recommendation.py     # Main end-to-end executable producing all FYP tables & plots
└── README.md                 # Module documentation and usage instructions
```

### Purpose and Major Functions of Each File
1. **`recommendation/config.py`**:
   - Stores paths to Model 1 outputs, Model 2 outputs, and raw catalog.
   - Defines hyperparameters: $K=10$, $\alpha=1.0$, $\beta=1.0$, default trust values ($0.75, 0.50$), risk penalties.
2. **`recommendation/candidate_generator.py`**:
   - `build_item_catalog()`: Extracts unique `(product_id, category_code, brand, price)` mappings.
   - `generate_session_candidates()`: Combines in-session and category affinity candidate pools.
3. **`recommendation/trust_integrator.py`**:
   - `load_model_outputs()`: Loads Model 1 probabilities and Model 2 trust scores.
   - `integrate_session_candidates()`: Joins candidates with Model 1 intent and Model 2 brand trust.
   - `apply_trust_fallbacks()`: Handles unproven and unknown brands.
4. **`recommendation/ranking.py`**:
   - `compute_trust_aware_scores()`: Applies the final trust-aware ranking formula.
   - `compute_baseline_scores()`: Computes Purchase-Only, Popularity, and Hard-Filter baseline ranks.
5. **`recommendation/evaluation.py`**:
   - `calculate_recommendation_metrics()`: Computes Precision@K, Recall@K, NDCG@K, HitRate@K.
   - `calculate_trust_safety_metrics()`: Computes Average Trust@K, High-Risk Exposure Rate.
6. **`recommendation/run_recommendation.py`**:
   - Single command runner that executes the entire recommendation pipeline, runs sensitivity analysis, and saves all tables and plots.

---

## 15. Existing Files to Modify

> [!NOTE]
> To preserve the integrity of the completed ML models, **NO files inside `behavioral-analysis/` or `user-intent/` will be altered.**

- **New files only:** All recommendation logic is strictly contained inside the new `recommendation/` folder.
- **Root README / Entrypoint (Optional):** A high-level runner or link in the root directory can optionally be provided to run the full FYP pipeline end-to-end.

---

## 16. End-to-End Execution Sequence

```text
STEP 1: Load precomputed Model 1 outputs
        (user-intent/data/session_purchase_probabilities.csv)
STEP 2: Load precomputed Model 2 outputs
        (behavioral-analysis/data/processed/final_brand_trust_dataset.csv)
STEP 3: Extract/load lightweight product catalog from clickstream data
STEP 4: Generate candidate items for test sessions
STEP 5: Integrate candidates -> Map product_id to brand
STEP 6: Attach purchase_probability (Model 1) and trust_score / risk_level (Model 2)
STEP 7: Apply fallback policies for unproven / unknown brands
STEP 8: Compute Base Relevance scores R(s, p)
STEP 9: Compute Final Trust-Aware scores S(s, p)
STEP 10: Sort and generate Top-K recommendations per session
STEP 11: Execute Baselines (Popularity, Purchase-Only, Hard-Filter) on identical candidates
STEP 12: Evaluate all 4 models on Utility (NDCG, HitRate) and Trust/Safety (ATS, HRER)
STEP 13: Execute Sensitivity Analysis across β in [0.0, 0.5, 1.0, 1.5, 2.0]
STEP 14: Save CSV result tables, Before vs After demonstration cases, and comparison plots
```

---

## 17. Computational & Memory Considerations

- **No Retraining:** Neither Model 1 nor Model 2 will be retrained during recommendation runs. Their precomputed `.pkl` artifacts and `.csv` datasets are loaded directly.
- **Catalog Caching:** The product catalog `(product_id, category_code, brand, price)` will be extracted once and cached to `recommendation/data/product_catalog.csv` (~2–5 MB), avoiding repeated scanning of the 5.66 GB raw dataset.
- **Session Evaluation Batching:** Recommendations and evaluations will be executed over the verified test sessions (1,069 sessions), running in less than 15 seconds on a standard laptop CPU.
- **Memory Footprint:** Peak RAM usage during recommendation generation will remain below 300 MB.

---

## 18. FYP Deliverable Outputs

The module will generate concrete artifacts for the dissertation and presentation:

### A. Summary Tables (CSV format)
1. `model_comparison_table.csv`: Compares Popularity vs Purchase-Only vs Hard-Filter vs Proposed Trust-Aware on:
   - HitRate@5, HitRate@10
   - NDCG@5, NDCG@10
   - Average Trust@5, Average Trust@10
   - High-Risk Exposure Rate@5, High-Risk Exposure Rate@10
2. `sensitivity_analysis_beta.csv`: Metric progression as $\beta$ varies from $0.0$ to $2.0$.
3. `before_after_case_studies.csv`: 5 distinct user session case studies detailing item rank shifts.

### B. Visualizations (PNG format, high-resolution 300 DPI)
1. `trust_vs_utility_tradeoff.png`: Two-axis plot illustrating NDCG@K vs Average Trust@K across $\beta$.
2. `high_risk_exposure_comparison.png`: Bar chart demonstrating reduction in high-risk recommendations across baselines.
3. `rank_displacement_distribution.png`: Histogram showing position demotion of low-trust products.

---

## 19. Risks & Design Decisions Requiring User Approval

| Decision Item | Options | Proposed Recommendation | Classification |
| :--- | :--- | :--- | :--- |
| **Candidate Generation Strategy** | A: In-session only<br>B: Global collaborative filtering<br>C: In-session + Category Co-occurrence | **Option C (In-session + Category Co-occurrence)** — Guarantees sufficient candidates without memory overload. | `[RECOMMENDED]` |
| **Trust Sensitivity ($\beta$)** | Fixed $\beta = 1.0$ vs Parametric sweep | **Include sweep $\beta \in [0.0, 2.0]$ with default $\beta = 1.0$** — Essential for the FYP research contribution. | `[RECOMMENDED]` |
| **Unscored Brand Trust Fallback** | 0.0 (punish) vs 1.0 (optimistic) vs 0.75 (neutral prior) | **0.75 (Conservative Neutral Prior)** — Avoids unfairly blocking smaller legitimate merchants. | `[RECOMMENDED]` |
| **Unknown Brand Fallback** | 0.0 vs 0.50 | **0.50 (Medium Risk)** — Reflects unbranded listing uncertainty. | `[RECOMMENDED]` |
| **High-Risk Brand Policy** | A: Continuous soft penalty ($\Psi = 0.20$)<br>B: Binary hard exclusion ($\Psi = 0.0$) | **Provide both:** Soft penalty as proposed model, Hard-Filter as Baseline 3 for direct thesis comparison. | `[RECOMMENDED]` |
| **Recommendation Cutoff ($K$)** | $K=5$ vs $K=10$ | **Report both $K=5$ and $K=10$** in evaluation tables. | `[RECOMMENDED]` |

---

## Confirmation / Next Steps

This concludes **Phase A**. In accordance with your strict instructions:
- **No implementation code has been written.**
- **Completed models have not been modified.**
- **Execution is halted awaiting your explicit review and confirmation.**
