# ============================================================
# train_purchase_model.py
# Standalone Purchase Prediction Pipeline & Model Serialization
# ============================================================

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PATH RESOLUTION
# ------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent

MODELS_DIR = CURRENT_DIR / "models"
DATA_DIR = CURRENT_DIR / "data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Candidate locations for 2019-Oct.csv
CANDIDATE_PATHS = [
    WORKSPACE_ROOT / "2019-Oct.csv",
    CURRENT_DIR / ".." / "2019-Oct.csv",
    Path("2019-Oct.csv").resolve()
]

RAW_DATA_PATH = None
for p in CANDIDATE_PATHS:
    if p.exists():
        RAW_DATA_PATH = str(p.resolve())
        break


# ------------------------------------------------------------
# 2. FEATURE SPECIFICATION
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "event_count",
    "view_count",
    "cart_count",
    "unique_products",
    "unique_categories",
    "unique_brands",
    "average_price",
    "max_price",
    "min_price",
    "session_duration",
    "hour",
    "day_of_week",
    "is_weekend",
    "last_action_1",
    "last_action_2",
    "last_action_3",
    "last_action_4",
    "last_action_5"
]


# ------------------------------------------------------------
# 3. DATA LOADING & PREPROCESSING
# ------------------------------------------------------------

def load_and_preprocess(raw_path=RAW_DATA_PATH, nrows=100000):
    if raw_path is None or not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Dataset not found at: {raw_path}. Please place 2019-Oct.csv in the workspace root."
        )

    print(f"Loading {nrows:,} rows from: {raw_path}...")
    df = pd.read_csv(raw_path, nrows=nrows)
    print(f"Raw rows loaded: {len(df):,}")

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing brand
    df["brand"] = df["brand"].fillna("unknown")

    # Drop category_code if present
    if "category_code" in df.columns:
        df = df.drop(columns=["category_code"])

    # Parse timestamps
    df["event_time"] = pd.to_datetime(df["event_time"])

    # Sort chronologically within sessions
    df = df.sort_values(["user_session", "event_time"]).reset_index(drop=True)

    # Create session-level purchase label
    session_labels = (
        df.groupby("user_session")["event_type"]
        .apply(lambda events: int("purchase" in events.values))
        .reset_index(name="purchase")
    )

    # Remove purchase events (predict before purchase)
    df_no_purchase = df[df["event_type"] != "purchase"].copy()

    # Filter sessions with at least 3 actions
    session_action_counts = df_no_purchase.groupby("user_session").size()
    valid_sessions = session_action_counts[session_action_counts >= 3].index
    df_valid = df_no_purchase[df_no_purchase["user_session"].isin(valid_sessions)].copy()

    # Session-level aggregated features
    session_features = (
        df_valid.groupby("user_session")
        .agg(
            event_count=("event_type", "count"),
            view_count=("event_type", lambda x: (x == "view").sum()),
            cart_count=("event_type", lambda x: (x == "cart").sum()),
            unique_products=("product_id", "nunique"),
            unique_categories=("category_id", "nunique"),
            unique_brands=("brand", "nunique"),
            average_price=("price", "mean"),
            max_price=("price", "max"),
            min_price=("price", "min"),
            session_start=("event_time", "min"),
            session_end=("event_time", "max"),
            user_id=("user_id", "first")
        )
        .reset_index()
    )

    session_features["session_duration"] = (
        session_features["session_end"] - session_features["session_start"]
    ).dt.total_seconds()

    session_features["hour"] = session_features["session_start"].dt.hour
    session_features["day_of_week"] = session_features["session_start"].dt.dayofweek
    session_features["is_weekend"] = (session_features["day_of_week"] >= 5).astype(int)

    # Merge labels
    session_features = session_features.merge(session_labels, on="user_session", how="left")
    session_features["purchase"] = session_features["purchase"].fillna(0).astype(int)

    num_cols = session_features.select_dtypes(include=np.number).columns
    session_features[num_cols] = session_features[num_cols].fillna(0)

    # Recent 5 actions representation
    N = 5
    action_data = df_valid.sort_values(["user_session", "event_time"]).copy()
    event_mapping = {"view": 1, "cart": 2}
    action_data["event_code"] = action_data["event_type"].map(event_mapping).fillna(0).astype(int)

    recent_actions = action_data.groupby("user_session").tail(N).copy()
    recent_actions["action_position"] = recent_actions.groupby("user_session").cumcount()

    recent_features = recent_actions.pivot(
        index="user_session",
        columns="action_position",
        values="event_code"
    )
    recent_features.columns = [f"last_action_{i+1}" for i in recent_features.columns]
    recent_features = recent_features.reset_index()

    for i in range(1, N + 1):
        col = f"last_action_{i}"
        if col not in recent_features.columns:
            recent_features[col] = 0

    recent_features = recent_features[["user_session"] + [f"last_action_{i}" for i in range(1, N + 1)]].fillna(0)
    action_cols = [f"last_action_{i}" for i in range(1, N + 1)]
    recent_features[action_cols] = recent_features[action_cols].astype(int)

    # Hybrid representation
    hybrid_df = session_features.merge(recent_features, on="user_session", how="inner")
    print(f"Hybrid feature matrix shape: {hybrid_df.shape}")
    return hybrid_df


# ------------------------------------------------------------
# 4. MODEL TRAINING & STACKING ENSEMBLE
# ------------------------------------------------------------

def train_and_evaluate(hybrid_df):
    X = hybrid_df[FEATURE_COLUMNS].copy()
    y = hybrid_df["purchase"].copy()
    sessions = hybrid_df["user_session"].copy()

    X_train, X_test, y_train, y_test, sessions_train, sessions_test = train_test_split(
        X, y, sessions, test_size=0.10, stratify=y, random_state=42
    )

    print(f"Training set: {len(X_train):,}, Test set: {len(X_test):,}")

    # Base Model 1: LightGBM
    print("Training LightGBM...")
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)

    # Base Model 2: XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    # Base Model 3: Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 5-fold Out-Of-Fold predictions for meta learner
    print("Generating 5-fold Out-Of-Fold predictions for Stacking...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_lgbm = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_rf = np.zeros(len(X_train))

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        m_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
        m_lgbm.fit(X_tr, y_tr)
        oof_lgbm[val_idx] = m_lgbm.predict_proba(X_val)[:, 1]

        m_xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss')
        m_xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]

        m_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        m_rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = m_rf.predict_proba(X_val)[:, 1]

    # Fit Meta Model
    print("Training Meta Model (Logistic Regression)...")
    meta_X_train = pd.DataFrame({"LightGBM": oof_lgbm, "XGBoost": oof_xgb, "RandomForest": oof_rf})
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_X_train, y_train)

    # Predictions on test set
    lgbm_test_prob = lgbm.predict_proba(X_test)[:, 1]
    xgb_test_prob = xgb.predict_proba(X_test)[:, 1]
    rf_test_prob = rf.predict_proba(X_test)[:, 1]

    meta_X_test = pd.DataFrame({"LightGBM": lgbm_test_prob, "XGBoost": xgb_test_prob, "RandomForest": rf_test_prob})
    stack_prob = meta_model.predict_proba(meta_X_test)[:, 1]

    # Threshold optimization
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_thresh = 0.50
    best_f1 = -1.0
    for t in thresholds:
        pred = (stack_prob >= t).astype(int)
        score = f1_score(y_test, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    final_pred = (stack_prob >= best_thresh).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, final_pred)),
        "precision": float(precision_score(y_test, final_pred, zero_division=0)),
        "recall": float(recall_score(y_test, final_pred, zero_division=0)),
        "f1": float(f1_score(y_test, final_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, stack_prob)),
        "auc_pr": float(average_precision_score(y_test, stack_prob)),
        "best_threshold": float(best_thresh)
    }

    print("\n" + "=" * 60)
    print("STACKING MODEL TEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:16s}: {v:.4f}")

    # Save artifacts
    print("\nSaving model artifacts to:", str(MODELS_DIR))
    joblib.dump(lgbm, MODELS_DIR / "purchase_lgbm.pkl")
    joblib.dump(xgb, MODELS_DIR / "purchase_xgb.pkl")
    joblib.dump(rf, MODELS_DIR / "purchase_rf.pkl")
    joblib.dump(meta_model, MODELS_DIR / "purchase_stacking_meta.pkl")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "best_threshold": float(best_thresh),
        "metrics": metrics
    }
    with open(MODELS_DIR / "purchase_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Save test predictions DataFrame
    test_results_df = pd.DataFrame({
        "user_session": sessions_test.values,
        "actual_purchase": y_test.values,
        "lgbm_prob": lgbm_test_prob,
        "xgb_prob": xgb_test_prob,
        "rf_prob": rf_test_prob,
        "purchase_probability": stack_prob,
        "predicted_label": final_pred
    })
    test_results_path = DATA_DIR / "purchase_prediction_test_results.csv"
    test_results_df.to_csv(test_results_path, index=False)
    print(f"Saved test predictions to: {test_results_path}")

    # Generate full dataset session purchase probabilities
    print("\nGenerating session purchase probabilities for all sessions...")
    all_lgbm = lgbm.predict_proba(X)[:, 1]
    all_xgb = xgb.predict_proba(X)[:, 1]
    all_rf = rf.predict_proba(X)[:, 1]
    all_meta_X = pd.DataFrame({"LightGBM": all_lgbm, "XGBoost": all_xgb, "RandomForest": all_rf})
    all_stack_prob = meta_model.predict_proba(all_meta_X)[:, 1]

    all_sessions_df = pd.DataFrame({
        "user_session": sessions.values,
        "actual_purchase": y.values,
        "purchase_probability": all_stack_prob,
        "predicted_purchase": (all_stack_prob >= best_thresh).astype(int)
    })
    all_sessions_path = DATA_DIR / "session_purchase_probabilities.csv"
    all_sessions_df.to_csv(all_sessions_path, index=False)
    print(f"Saved session purchase probabilities to: {all_sessions_path}")

    return metadata, test_results_df, all_sessions_df


if __name__ == "__main__":
    print("=" * 70)
    print("PURCHASE PREDICTION MODEL - STANDALONE RUNNER")
    print("=" * 70)
    hybrid_df = load_and_preprocess(nrows=100000)
    metadata, test_results_df, all_sessions_df = train_and_evaluate(hybrid_df)
    print("\nPurchase Prediction Model execution completed successfully!")
