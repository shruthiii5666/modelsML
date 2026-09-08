# ============================================================
# config.py
# Central Configuration for Behavioral Analysis and Brand Risk
# ============================================================

import os
from pathlib import Path

# ------------------------------------------------------------
# 1. BASE DIRECTORIES & PATH RESOLUTION
# ------------------------------------------------------------

# Base directory for behavioral-analysis
SRC_DIR = Path(__file__).resolve().parent
BEHAVIORAL_DIR = SRC_DIR.parent
WORKSPACE_ROOT = BEHAVIORAL_DIR.parent

# Candidate locations for raw 2019-Oct.csv
CANDIDATE_RAW_PATHS = [
    WORKSPACE_ROOT / "2019-Oct.csv",
    BEHAVIORAL_DIR / "data" / "raw" / "2019-Oct.csv",
    BEHAVIORAL_DIR / ".." / "2019-Oct.csv",
    Path("2019-Oct.csv").resolve(),
    Path("data/raw/2019-Oct.csv").resolve()
]

RAW_DATA_PATH = str(WORKSPACE_ROOT / "2019-Oct.csv")
for p in CANDIDATE_RAW_PATHS:
    if p.exists():
        RAW_DATA_PATH = str(p.resolve())
        break

# Processed, output, and model directories inside behavioral-analysis
DATA_DIR = BEHAVIORAL_DIR / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = BEHAVIORAL_DIR / "models"

# Ensure directories exist automatically
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. PROCESSED & INTERMEDIATE DATA PATHS
# ------------------------------------------------------------

CLEANED_DATA_PATH = str(DATA_PROCESSED_DIR / "cleaned_events.csv")
SESSION_DATA_PATH = str(DATA_PROCESSED_DIR / "session_features.csv")
SESSION_BRAND_DATA_PATH = str(DATA_PROCESSED_DIR / "session_brand_features.csv")
BRAND_DATA_PATH = str(DATA_PROCESSED_DIR / "brand_features.csv")
BRAND_CATEGORY_DATA_PATH = str(DATA_PROCESSED_DIR / "brand_category_features.csv")
FINAL_DATA_PATH = str(DATA_PROCESSED_DIR / "final_brand_dataset.csv")

# Model preparation and anomaly detection paths
MODEL_READY_DATA_PATH = str(DATA_PROCESSED_DIR / "model_ready_features.csv")
MODEL_FEATURE_MATRIX_PATH = str(DATA_PROCESSED_DIR / "model_feature_matrix.csv")
ANOMALY_RESULTS_PATH = str(DATA_PROCESSED_DIR / "anomaly_results.csv")
TRUST_DATASET_PATH = str(DATA_PROCESSED_DIR / "final_brand_trust_dataset.csv")


# ------------------------------------------------------------
# 3. OUTPUT PATHS
# ------------------------------------------------------------

BRAND_SCORES_PATH = str(DATA_OUTPUT_DIR / "brand_scores.csv")


# ------------------------------------------------------------
# 4. MODEL & SCALER ARTIFACT PATHS
# ------------------------------------------------------------

MODEL_PATH = str(MODELS_DIR / "isolation_forest.pkl")
SCALER_PATH = str(MODELS_DIR / "feature_scaler.pkl")
FEATURE_LIST_PATH = str(MODELS_DIR / "model_feature_columns.txt")


# ------------------------------------------------------------
# 5. LARGE DATASET CHUNKING SETTINGS
# ------------------------------------------------------------

CHUNK_SIZE = 500_000

# Number of chunks used in prototype (3 chunks * 500,000 = ~1.5 million rows)
DEFAULT_MAX_CHUNKS = 3


# ------------------------------------------------------------
# 6. REQUIRED COLUMNS
# ------------------------------------------------------------

REQUIRED_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session"
]


# ------------------------------------------------------------
# 7. EXPECTED EVENT TYPES
# ------------------------------------------------------------

VALID_EVENT_TYPES = [
    "view",
    "cart",
    "purchase"
]


# ------------------------------------------------------------
# 8. MINIMUM EVIDENCE THRESHOLDS
# ------------------------------------------------------------

MIN_VIEWS = 20
MIN_SESSIONS = 10
MIN_USERS = 5


# ------------------------------------------------------------
# 9. ISOLATION FOREST SETTINGS
# ------------------------------------------------------------

CONTAMINATION = 0.05
RANDOM_STATE = 42
N_ESTIMATORS = 300


# ------------------------------------------------------------
# 10. SUSPICIOUSNESS SCORE THRESHOLDS
# ------------------------------------------------------------

LOW_RISK_THRESHOLD = 0.33
HIGH_RISK_THRESHOLD = 0.67


# ------------------------------------------------------------
# 11. DATA TYPE SETTINGS
# ------------------------------------------------------------

DATETIME_COLUMN = "event_time"

CATEGORICAL_COLUMNS = [
    "event_type",
    "brand",
    "category_code"
]

ID_COLUMNS = [
    "product_id",
    "category_id",
    "user_id",
    "user_session"
]

NUMERIC_COLUMNS = [
    "price"
]