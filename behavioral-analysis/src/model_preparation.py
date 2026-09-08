# ============================================================
# model_preparation.py
# Phase 7: Evidence Filtering + Model Feature Preparation
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler

from src.config import (
    FINAL_DATA_PATH,
    MODEL_READY_DATA_PATH,
    MODEL_FEATURE_MATRIX_PATH,
    SCALER_PATH,
    FEATURE_LIST_PATH
)


# ============================================================
# FEATURES USED FOR ISOLATION FOREST
# ============================================================

COUNT_FEATURES = [
    "total_events",
    "total_views",
    "total_carts",
    "total_purchases",
    "unique_sessions",
    "unique_users",
    "unique_products",
    "events_per_session",
    "views_per_session",
    "purchases_per_session",
    "events_per_user",
    "purchases_per_user",
    "events_per_product",
]


RATE_FEATURES = [
    "view_to_cart_rate",
    "cart_to_purchase_rate",
    "view_to_purchase_rate",
    "purchase_event_rate",
    "cart_event_rate",
    "view_event_rate",
]


DEVIATION_FEATURES = [
    "view_to_cart_deviation",
    "cart_to_purchase_deviation",
    "view_to_purchase_deviation",
    "purchase_event_deviation",
    "cart_event_deviation",
    "view_event_deviation",
]


RELATIVE_FEATURES = [
    "price_relative_ratio",
    "event_share_in_category",
    "view_share_in_category",
    "cart_share_in_category",
    "purchase_share_in_category",
]


# ============================================================
# CHECK INPUT DATASET
# ============================================================

def check_input_dataset():
    """
    Check whether the Phase 6 dataset exists.
    """

    if not os.path.exists(FINAL_DATA_PATH):

        raise FileNotFoundError(
            "\nPhase 6 dataset not found!\n"
            f"Expected file:\n{os.path.abspath(FINAL_DATA_PATH)}\n\n"
            "Run Phase 6 before starting Phase 7."
        )

    print("\n" + "=" * 60)
    print("PHASE 7 INPUT CHECK")
    print("=" * 60)

    print(f"Input dataset: {FINAL_DATA_PATH}")

    return True


# ============================================================
# LOAD PHASE 6 DATASET
# ============================================================

def load_final_dataset():
    """
    Load the 4,106-row Phase 6 dataset.

    This function does NOT read the raw 2019-Oct.csv file.
    """

    check_input_dataset()

    df = pd.read_csv(FINAL_DATA_PATH)

    print(f"Loaded dataset shape: {df.shape}")

    return df


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

def validate_required_columns(df):

    required_columns = [
        "brand",
        "category_code",
        "sufficient_evidence",
    ]

    required_columns.extend(COUNT_FEATURES)
    required_columns.extend(RATE_FEATURES)
    required_columns.extend(DEVIATION_FEATURES)
    required_columns.extend(RELATIVE_FEATURES)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns for Phase 7:\n"
            f"{missing_columns}"
        )

    print("\nRequired columns check: PASSED")


# ============================================================
# EVIDENCE FILTERING
# ============================================================

def filter_sufficient_evidence(df):
    """
    Keep only brand-category combinations with sufficient evidence.

    sufficient_evidence =

        sufficient_views
        AND sufficient_sessions
        AND sufficient_users
    """

    print("\n" + "=" * 60)
    print("EVIDENCE FILTERING")
    print("=" * 60)

    before_count = len(df)

    print(f"Before filtering: {before_count:,}")

    print("\nEvidence distribution:")

    print(
        df["sufficient_evidence"]
        .value_counts()
        .sort_index()
    )

    df_model = df[
        df["sufficient_evidence"] == 1
    ].copy()

    after_count = len(df_model)

    removed_count = before_count - after_count

    print(f"\nAfter filtering: {after_count:,}")
    print(f"Removed: {removed_count:,}")

    if before_count > 0:

        percentage = (
            after_count / before_count
        ) * 100

        print(
            f"Retained: {percentage:.2f}%"
        )

    return df_model


# ============================================================
# HANDLE NUMERIC VALUES
# ============================================================

def prepare_numeric_values(df, feature_columns):
    """
    Convert selected features to numeric and replace
    invalid infinite values.
    """

    df = df.copy()

    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df[feature_columns] = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    return df


# ============================================================
# LOG TRANSFORMATION
# ============================================================

def apply_log_transform(df):
    """
    Apply log1p transformation to count/volume features.

    This reduces the effect of highly skewed values such as
    very popular brands receiving thousands of events.
    """

    df = df.copy()

    transformed_columns = []

    for column in COUNT_FEATURES:

        if column not in df.columns:
            continue

        # Count-like features should be non-negative.
        # Small negative values, if any, are clipped.
        df[column] = df[column].clip(lower=0)

        transformed_name = "log_" + column

        df[transformed_name] = np.log1p(
            df[column]
        )

        transformed_columns.append(
            transformed_name
        )

    print("\n" + "=" * 60)
    print("LOG TRANSFORMATION")
    print("=" * 60)

    print(
        f"Log-transformed features: "
        f"{len(transformed_columns)}"
    )

    for column in transformed_columns:
        print(f"  - {column}")

    return df, transformed_columns


# ============================================================
# BUILD MODEL FEATURE SET
# ============================================================

def build_model_features(df, log_columns):
    """
    Build the final feature matrix for Isolation Forest.

    Original count features are replaced by their log versions.
    """

    model_features = []

    # --------------------------------------------------------
    # Log-transformed volume features
    # --------------------------------------------------------

    model_features.extend(log_columns)

    # --------------------------------------------------------
    # Behavioral rates
    # --------------------------------------------------------

    model_features.extend(
        RATE_FEATURES
    )

    # --------------------------------------------------------
    # Category-relative deviations
    # --------------------------------------------------------

    model_features.extend(
        DEVIATION_FEATURES
    )

    # --------------------------------------------------------
    # Relative behavior
    # --------------------------------------------------------

    model_features.extend(
        RELATIVE_FEATURES
    )

    # Remove duplicates while preserving order
    model_features = list(
        dict.fromkeys(model_features)
    )

    missing_features = [
        column
        for column in model_features
        if column not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "\nMissing model features:\n"
            f"{missing_features}"
        )

    X = df[model_features].copy()

    return X, model_features


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(X):
    """
    Replace missing numeric values using median values.

    Median is preferred because the behavioral dataset
    contains legitimate extreme observations.
    """

    X = X.copy()

    missing_before = X.isnull().sum().sum()

    if missing_before > 0:

        print("\nMissing feature values detected:")
        print(
            X.isnull()
            .sum()
            .loc[
                lambda x: x > 0
            ]
        )

        for column in X.columns:

            if X[column].isnull().any():

                median_value = X[column].median()

                X[column] = X[column].fillna(
                    median_value
                )

    missing_after = X.isnull().sum().sum()

    print("\n" + "=" * 60)
    print("MISSING VALUE HANDLING")
    print("=" * 60)

    print(
        f"Missing values before: "
        f"{missing_before:,}"
    )

    print(
        f"Missing values after: "
        f"{missing_after:,}"
    )

    return X


# ============================================================
# ROBUST SCALING
# ============================================================

def scale_features(X):
    """
    Scale features using RobustScaler.

    RobustScaler uses median and IQR, making it less sensitive
    to extreme behavioral observations.
    """

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )

    print("\n" + "=" * 60)
    print("ROBUST SCALING")
    print("=" * 60)

    print(
        f"Features scaled: {X_scaled.shape[1]}"
    )

    return X_scaled, scaler


# ============================================================
# SAVE MODEL-READY DATASET
# ============================================================

def save_model_ready_dataset(
    df,
    X_scaled,
    model_features
):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset containing identifiers + scaled features
    # --------------------------------------------------------

    output_df = df[
        [
            "brand",
            "category_code",
            "sufficient_evidence",
        ]
    ].copy()

    for column in model_features:

        output_df[column] = X_scaled[column]

    output_df.to_csv(
        MODEL_READY_DATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Feature matrix only
    # --------------------------------------------------------

    X_scaled.to_csv(
        MODEL_FEATURE_MATRIX_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler_global,
        SCALER_PATH
    )

    # --------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------

    with open(
        FEATURE_LIST_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for feature in model_features:
            file.write(feature + "\n")

    print("\n" + "=" * 60)
    print("PHASE 7 FILES SAVED")
    print("=" * 60)

    print(
        f"\nModel-ready dataset:\n"
        f"{MODEL_READY_DATA_PATH}"
    )

    print(
        f"\nFeature matrix:\n"
        f"{MODEL_FEATURE_MATRIX_PATH}"
    )

    print(
        f"\nScaler:\n"
        f"{SCALER_PATH}"
    )

    print(
        f"\nFeature list:\n"
        f"{FEATURE_LIST_PATH}"
    )


# ============================================================
# MAIN PHASE 7 PIPELINE
# ============================================================

def run_phase_7():

    global scaler_global

    print("\n")
    print("=" * 60)
    print("PHASE 7")
    print("EVIDENCE FILTERING + MODEL FEATURE PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load Phase 6 dataset
    # --------------------------------------------------------

    df = load_final_dataset()

    # --------------------------------------------------------
    # 2. Validate columns
    # --------------------------------------------------------

    validate_required_columns(df)

    # --------------------------------------------------------
    # 3. Evidence filtering
    # --------------------------------------------------------

    df_model = filter_sufficient_evidence(df)

    if len(df_model) == 0:

        raise ValueError(
            "\nNo observations remain after evidence filtering."
        )

    # --------------------------------------------------------
    # 4. Prepare numeric values
    # --------------------------------------------------------

    all_original_features = (
        COUNT_FEATURES
        + RATE_FEATURES
        + DEVIATION_FEATURES
        + RELATIVE_FEATURES
    )

    df_model = prepare_numeric_values(
        df_model,
        all_original_features
    )

    # --------------------------------------------------------
    # 5. Log transform count features
    # --------------------------------------------------------

    df_model, log_columns = apply_log_transform(
        df_model
    )

    # --------------------------------------------------------
    # 6. Build feature matrix
    # --------------------------------------------------------

    X, model_features = build_model_features(
        df_model,
        log_columns
    )

    print("\n" + "=" * 60)
    print("MODEL FEATURE SELECTION")
    print("=" * 60)

    print(
        f"Number of model features: "
        f"{len(model_features)}"
    )

    print("\nSelected features:")

    for index, feature in enumerate(
        model_features,
        start=1
    ):

        print(
            f"{index:02d}. {feature}"
        )

    # --------------------------------------------------------
    # 7. Handle missing values
    # --------------------------------------------------------

    X = handle_missing_values(X)

    # --------------------------------------------------------
    # 8. Robust scaling
    # --------------------------------------------------------

    X_scaled, scaler = scale_features(X)

    scaler_global = scaler

    # --------------------------------------------------------
    # 9. Save outputs
    # --------------------------------------------------------

    save_model_ready_dataset(
        df_model,
        X_scaled,
        model_features
    )

    # --------------------------------------------------------
    # 10. Final checks
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 7 FINAL CHECK")
    print("=" * 60)

    print(
        f"\nFinal model dataset shape: "
        f"{X_scaled.shape}"
    )

    print(
        f"Rows used for model training: "
        f"{len(X_scaled):,}"
    )

    print(
        f"Features used: "
        f"{len(model_features)}"
    )

    print(
        "\nAny missing values:",
        X_scaled.isnull().sum().sum()
    )

    print(
        "Any infinite values:",
        np.isinf(X_scaled.to_numpy()).sum()
    )

    print("\nFirst 5 scaled rows:")

    print(
        X_scaled.head()
    )

    print("\n" + "=" * 60)
    print("PHASE 7 COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_phase_7()