# ============================================================
# recommendation/config.py
# Configuration for Trust-Aware Recommendation Module
# ============================================================

import os
from pathlib import Path

# ------------------------------------------------------------
# 1. DIRECTORY & FILE PATHS
# ------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent

DATA_DIR = CURRENT_DIR / "data"
PLOTS_DIR = CURRENT_DIR / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate locations for raw 2019-Oct.csv
CANDIDATE_RAW_PATHS = [
    WORKSPACE_ROOT / "2019-Oct.csv",
    CURRENT_DIR / ".." / "2019-Oct.csv",
    Path("2019-Oct.csv").resolve()
]

RAW_CLICKSTREAM_PATH = str(WORKSPACE_ROOT / "2019-Oct.csv")
for p in CANDIDATE_RAW_PATHS:
    if p.exists():
        RAW_CLICKSTREAM_PATH = str(p.resolve())
        break

# Model 1 Inputs (Purchase Intent)
MODEL1_DIR = WORKSPACE_ROOT / "user-intent" / "data"
MODEL1_PROBABILITIES_PATH = str(MODEL1_DIR / "session_purchase_probabilities.csv")
MODEL1_TEST_RESULTS_PATH = str(MODEL1_DIR / "purchase_prediction_test_results.csv")

# Model 2 Inputs (Brand Risk & Trust)
MODEL2_PROCESSED_DIR = WORKSPACE_ROOT / "behavioral-analysis" / "data" / "processed"
MODEL2_OUTPUT_DIR = WORKSPACE_ROOT / "behavioral-analysis" / "data" / "output"
MODEL2_TRUST_DATASET_PATH = str(MODEL2_PROCESSED_DIR / "final_brand_trust_dataset.csv")
MODEL2_BRAND_SCORES_PATH = str(MODEL2_OUTPUT_DIR / "brand_scores.csv")

# Recommendation Cache and Output Paths
PRODUCT_CATALOG_PATH = str(DATA_DIR / "product_catalog.csv")
CANDIDATE_CACHE_PATH = str(DATA_DIR / "recommendation_candidates.csv")
TOPK_TRUST_AWARE_PATH = str(DATA_DIR / "topk_recommendations_trust_aware.csv")
TOPK_PURCHASE_ONLY_PATH = str(DATA_DIR / "topk_recommendations_purchase_only.csv")
MODEL_COMPARISON_PATH = str(DATA_DIR / "model_comparison_table.csv")
SENSITIVITY_ANALYSIS_PATH = str(DATA_DIR / "sensitivity_analysis_beta.csv")
BEFORE_AFTER_CASES_PATH = str(DATA_DIR / "before_after_case_studies.csv")

# Plot Output Paths
PLOT_TRADEOFF_PATH = str(PLOTS_DIR / "trust_vs_utility_tradeoff.png")
PLOT_HIGH_RISK_PATH = str(PLOTS_DIR / "high_risk_exposure_comparison.png")
PLOT_RANK_DISPLACEMENT_PATH = str(PLOTS_DIR / "rank_displacement_distribution.png")


# ------------------------------------------------------------
# 2. RECOMMENDATION & RANKING HYPERPARAMETERS
# ------------------------------------------------------------

DEFAULT_K = 10
EVALUATION_K_VALUES = [5, 10]

# Ranking formula weights:
# Score = Relevance * (1.0 + ALPHA * Purchase_Prob) * (Trust_Score ** BETA) * Risk_Penalty
ALPHA = 1.0
BETA = 1.0
BETA_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]

# Discrete Risk Penalty Multipliers Psi(b)
RISK_PENALTIES = {
    "Low": 1.00,
    "Medium": 0.75,
    "High": 0.20
}

# Empirical Fallback Trust Priors
DEFAULT_UNSCORED_TRUST = 0.75   # Unscored / low-evidence brands
DEFAULT_UNKNOWN_TRUST = 0.50    # Missing / unbranded products
DEFAULT_UNSCORED_RISK = "Medium"
DEFAULT_UNKNOWN_RISK = "Medium"


# ------------------------------------------------------------
# 3. CANDIDATE GENERATION SETTINGS
# ------------------------------------------------------------

IN_SESSION_CART_RELEVANCE = 1.00
IN_SESSION_VIEW_RELEVANCE = 0.80
CATEGORY_AFFINITY_MIN_RELEVANCE = 0.20
CATEGORY_AFFINITY_MAX_RELEVANCE = 0.60
MAX_CATEGORY_CANDIDATES_PER_SESSION = 25
SAMPLE_CATALOG_ROWS = 200_000   # Sufficient to extract rich catalog and item co-occurrences
