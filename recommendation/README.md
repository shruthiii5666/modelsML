# Trust-Aware E-Commerce Recommendation Module

This module integrates the completed **User Purchase Prediction Model** (Model 1) and **Brand Risk Model** (Model 2) to generate trust-aware, safety-optimized product recommendations.

---

## Architecture Overview

```text
       Clickstream Session Interactions (2019-Oct.csv)
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │   recommendation/candidate_generator.py   │
        │  In-Session Items + Category Co-occurrence│
        └─────────────────────┬─────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│    Model 1 Output (Intent)  │   │  Model 2 Output (Brand Risk)│
│session_purchase_            │   │final_brand_trust_dataset.csv│
│  probabilities.csv          │   │                             │
│Key: user_session            │   │Key: brand                   │
│Value: purchase_probability  │   │Value: trust_score, risk     │
└──────────────┬──────────────┘   └──────────────┬──────────────┘
               │                                 │
               └──────────────┬──────────────────┘
                              ▼
        ┌───────────────────────────────────────────┐
        │    recommendation/trust_integrator.py     │
        │  Assembles recommendation_candidates_df   │
        │  Applies Fallback Policies (Unknown/Prior)│
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │         recommendation/ranking.py         │
        │ S(s,p) = Relevance × (1+α·P) × Trust^β × Ψ│
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │        recommendation/evaluation.py       │
        │  HitRate@K, Precision@K, NDCG@K, ATS@K    │
        └───────────────────────────────────────────┘
```

---

## File Structure

```text
recommendation/
│
├── __init__.py
├── config.py                 # Hyperparameters, weights (alpha, beta), paths & fallback priors
├── candidate_generator.py    # Extracts lightweight catalog, in-session & category candidates
├── trust_integrator.py       # Joins Model 1 intent + Model 2 brand trust and applies fallbacks
├── ranking.py                # Computes Trust-Aware score & 3 comparative baselines
├── evaluation.py             # Computes HitRate, Precision, Recall, NDCG, ATS, High-Risk Exposure
├── run_recommendation.py     # Main end-to-end execution runner with visualization generation
├── README.md                 # Documentation
│
├── data/                     # Generated recommendation artifacts & tables
│   ├── product_catalog.csv
│   ├── recommendation_candidates.csv
│   ├── topk_recommendations_trust_aware.csv
│   ├── topk_recommendations_purchase_only.csv
│   ├── model_comparison_table.csv
│   ├── sensitivity_analysis_beta.csv
│   └── before_after_case_studies.csv
│
└── plots/                    # High-resolution (300 DPI) publication-ready plots
    ├── trust_vs_utility_tradeoff.png
    ├── high_risk_exposure_comparison.png
    └── rank_displacement_distribution.png
```

---

## Ranking Formula

$$S(s, p) = R(s, p) \times \left( 1.0 + \alpha \cdot P(\text{purchase} \mid s) \right) \times \left( \text{Trust}(b) \right)^\beta \times \Psi(b)$$

Where:
- $R(s, p) \in [0.1, 1.0]$: Base Relevance score (1.0 for cart, 0.8 for view, 0.20–0.60 for category affinity).
- $P(\text{purchase} \mid s) \in [0.0, 1.0]$: Purchase probability from Model 1 (Stacking Ensemble).
- $\text{Trust}(b) \in [0.0, 1.0]$: Brand trust score from Model 2 (Isolation Forest anomaly inversion).
- $\alpha \ge 0$: Purchase intent weight (default $= 1.0$).
- $\beta \ge 0$: Trust sensitivity parameter (default $= 1.0$; setting $\beta=0, \Psi=1.0$ yields the Purchase-Only baseline).
- $\Psi(b)$: Discrete risk demotion multiplier:
  - Low Risk: $1.00$
  - Medium Risk: $0.75$
  - High Risk: $0.20$

---

## How to Run

From the project root:

```bash
python recommendation/run_recommendation.py
```
