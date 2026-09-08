# ============================================================
# anomaly_model.py
# Phase 8: Isolation Forest Anomaly Detection
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest

from src.config import (
    CONTAMINATION,
    RANDOM_STATE,
    N_ESTIMATORS,
    MODEL_READY_DATA_PATH,
    MODEL_FEATURE_MATRIX_PATH,
    MODEL_PATH,
    ANOMALY_RESULTS_PATH
)


# ============================================================
# LOAD MODEL DATA
# ============================================================

def load_model_data():

    if not os.path.exists(MODEL_READY_DATA_PATH):

        raise FileNotFoundError(
            "\nModel-ready dataset not found.\n"
            "Run Phase 7 first."
        )

    if not os.path.exists(MODEL_FEATURE_MATRIX_PATH):

        raise FileNotFoundError(
            "\nFeature matrix not found.\n"
            "Run Phase 7 first."
        )

    metadata = pd.read_csv(
        MODEL_READY_DATA_PATH
    )

    X = pd.read_csv(
        MODEL_FEATURE_MATRIX_PATH
    )

    print("\n" + "=" * 60)
    print("PHASE 8 INPUT CHECK")
    print("=" * 60)

    print(
        f"Metadata shape: {metadata.shape}"
    )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    return metadata, X


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

def train_isolation_forest(X):

    print("\n" + "=" * 60)
    print("TRAINING ISOLATION FOREST")
    print("=" * 60)

    print(
        f"Number of observations: {len(X):,}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    print(
        f"Number of trees: {N_ESTIMATORS}"
    )

    print(
        f"Contamination: {CONTAMINATION}"
    )

    print(
        f"Random state: {RANDOM_STATE}"
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X)

    print("\nIsolation Forest training completed.")

    return model


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_anomaly_results(
    metadata,
    X,
    model
):

    print("\n" + "=" * 60)
    print("GENERATING ANOMALY SCORES")
    print("=" * 60)

    # --------------------------------------------------------
    # Isolation Forest prediction
    #
    # +1 = normal
    # -1 = anomaly
    # --------------------------------------------------------

    predictions = model.predict(X)

    # --------------------------------------------------------
    # decision_function
    #
    # Larger values = more normal
    # Smaller values = more anomalous
    # --------------------------------------------------------

    decision_scores = (
        model.decision_function(X)
    )

    # --------------------------------------------------------
    # Convert decision score into an anomaly-oriented score
    #
    # Lower decision score = more anomalous
    #
    # We normalize it later for interpretation.
    # --------------------------------------------------------

    anomaly_raw = -decision_scores

    # --------------------------------------------------------
    # Min-max normalization
    #
    # 0 = least anomalous in this dataset
    # 1 = most anomalous in this dataset
    # --------------------------------------------------------

    min_score = anomaly_raw.min()
    max_score = anomaly_raw.max()

    if max_score > min_score:

        anomaly_score = (
            (anomaly_raw - min_score)
            / (max_score - min_score)
        )

    else:

        anomaly_score = np.zeros(
            len(anomaly_raw)
        )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    results = metadata[
        [
            "brand",
            "category_code"
        ]
    ].copy()

    results["is_anomaly"] = (
        predictions == -1
    ).astype(int)

    results["decision_score"] = (
        decision_scores
    )

    results["anomaly_raw"] = (
        anomaly_raw
    )

    results["anomaly_score"] = (
        anomaly_score
    )

    return results


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved:")
    print(MODEL_PATH)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    results.to_csv(
        ANOMALY_RESULTS_PATH,
        index=False
    )

    print("\nAnomaly results saved:")
    print(ANOMALY_RESULTS_PATH)


# ============================================================
# ANALYZE RESULTS
# ============================================================

def analyze_results(results):

    print("\n" + "=" * 60)
    print("ANOMALY RESULT ANALYSIS")
    print("=" * 60)

    total = len(results)

    anomalies = (
        results["is_anomaly"] == 1
    ).sum()

    normal = (
        results["is_anomaly"] == 0
    ).sum()

    print(
        f"\nTotal observations: {total:,}"
    )

    print(
        f"Normal observations: {normal:,}"
    )

    print(
        f"Anomalous observations: {anomalies:,}"
    )

    if total > 0:

        print(
            f"Anomaly percentage: "
            f"{(anomalies / total) * 100:.2f}%"
        )

    print("\nAnomaly score statistics:")

    print(
        results["anomaly_score"]
        .describe()
    )

    print("\nTop 20 most anomalous brand-category combinations:")

    top_anomalies = (
        results
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        [
            [
                "brand",
                "category_code",
                "is_anomaly",
                "decision_score",
                "anomaly_score"
            ]
        ]
        .head(20)
    )

    print(
        top_anomalies.to_string(
            index=False
        )
    )


# ============================================================
# MAIN PHASE 8
# ============================================================

def run_phase_8():

    print("\n")
    print("=" * 60)
    print("PHASE 8")
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load Phase 7 data
    # --------------------------------------------------------

    metadata, X = load_model_data()

    # --------------------------------------------------------
    # 2. Train model
    # --------------------------------------------------------

    model = train_isolation_forest(X)

    # --------------------------------------------------------
    # 3. Generate anomaly scores
    # --------------------------------------------------------

    results = generate_anomaly_results(
        metadata,
        X,
        model
    )

    # --------------------------------------------------------
    # 4. Save model
    # --------------------------------------------------------

    save_model(model)

    # --------------------------------------------------------
    # 5. Save anomaly results
    # --------------------------------------------------------

    save_results(results)

    # --------------------------------------------------------
    # 6. Analyze
    # --------------------------------------------------------

    analyze_results(results)

    # --------------------------------------------------------
    # 7. Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 8 FINAL CHECK")
    print("=" * 60)

    print(
        f"Result shape: {results.shape}"
    )

    print(
        "Missing anomaly scores:",
        results["anomaly_score"].isnull().sum()
    )

    print(
        "Missing predictions:",
        results["is_anomaly"].isnull().sum()
    )

    print("\n" + "=" * 60)
    print("PHASE 8 COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_phase_8()