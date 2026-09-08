# ============================================================
# recommendation/trust_integrator.py
# Merges Model 1 Purchase Intent and Model 2 Brand Trust
# ============================================================

import os
import pandas as pd
import numpy as np

from recommendation.config import (
    MODEL1_PROBABILITIES_PATH,
    MODEL1_TEST_RESULTS_PATH,
    MODEL2_TRUST_DATASET_PATH,
    DEFAULT_UNSCORED_TRUST,
    DEFAULT_UNKNOWN_TRUST,
    DEFAULT_UNSCORED_RISK,
    DEFAULT_UNKNOWN_RISK
)


def load_purchase_intent_data(
    prob_path=MODEL1_PROBABILITIES_PATH,
    test_path=MODEL1_TEST_RESULTS_PATH
):
    """
    Load Model 1 session-level purchase probabilities.
    Prefers test results if available, else full session probability table.
    """
    if os.path.exists(test_path):
        print(f"Loading Model 1 test set predictions: {test_path}")
        df = pd.read_csv(test_path)
        return df[["user_session", "actual_purchase", "purchase_probability", "predicted_label"]].rename(
            columns={"predicted_label": "predicted_purchase"}
        )

    if os.path.exists(prob_path):
        print(f"Loading Model 1 session purchase probabilities: {prob_path}")
        df = pd.read_csv(prob_path)
        return df[["user_session", "actual_purchase", "purchase_probability", "predicted_purchase"]]

    raise FileNotFoundError(
        f"Model 1 outputs not found at {prob_path} or {test_path}. "
        "Run 'python user-intent/train_purchase_model.py' first."
    )


def load_brand_trust_data(trust_path=MODEL2_TRUST_DATASET_PATH):
    """
    Load Model 2 brand trust scores and risk bands.
    """
    if not os.path.exists(trust_path):
        raise FileNotFoundError(
            f"Model 2 trust dataset not found at: {trust_path}. "
            "Run 'python behavioral-analysis/main.py' first."
        )

    print(f"Loading Model 2 brand trust dataset: {trust_path}")
    df = pd.read_csv(trust_path)

    # Standardize brand
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()

    cols_to_keep = [
        "brand",
        "trust_score",
        "suspiciousness_score",
        "risk_level",
        "trust_category",
        "scoring_status"
    ]
    return df[cols_to_keep].drop_duplicates(subset=["brand"])


def integrate_candidate_pool(
    candidates_df,
    intent_df,
    trust_df,
    default_unscored_trust=DEFAULT_UNSCORED_TRUST,
    default_unknown_trust=DEFAULT_UNKNOWN_TRUST,
    default_unscored_risk=DEFAULT_UNSCORED_RISK,
    default_unknown_risk=DEFAULT_UNKNOWN_RISK
):
    """
    Assemble the master recommendation_candidates_df by linking:
    candidates_df + Model 1 (purchase_probability) + Model 2 (trust_score, risk_level)
    and applying approved fallback policies.
    """
    print(f"Integrating {len(candidates_df):,} candidate rows with ML model outputs...")

    df = candidates_df.copy()

    # Normalize brand in candidates
    df["brand"] = df["brand"].fillna("unknown").astype(str).str.strip().str.lower()
    df.loc[df["brand"] == "", "brand"] = "unknown"

    # 1. Join Model 1 Purchase Probabilities on user_session
    df = df.merge(
        intent_df[["user_session", "purchase_probability", "predicted_purchase"]],
        on="user_session",
        how="left"
    )

    # Handle sessions without explicit Model 1 probability (fallback to empirical base intent rate 0.05)
    df["purchase_probability"] = pd.to_numeric(df["purchase_probability"], errors="coerce").fillna(0.05).astype(float)
    df["predicted_purchase"] = pd.to_numeric(df["predicted_purchase"], errors="coerce").fillna(0).astype(int)

    # 2. Join Model 2 Brand Trust on brand
    df = df.merge(
        trust_df,
        on="brand",
        how="left"
    )

    # 3. Apply Fallback Policies
    # Unknown brands
    unknown_mask = (df["brand"] == "unknown")
    df.loc[unknown_mask, "trust_score"] = default_unknown_trust
    df.loc[unknown_mask, "suspiciousness_score"] = 1.0 - default_unknown_trust
    df.loc[unknown_mask, "risk_level"] = default_unknown_risk
    df.loc[unknown_mask, "trust_category"] = "Unknown Brand"
    df.loc[unknown_mask, "scoring_status"] = "unbranded_fallback"

    # Unscored / insufficient evidence brands (known name, but missing in Model 2)
    unscored_mask = df["trust_score"].isna()
    df.loc[unscored_mask, "trust_score"] = default_unscored_trust
    df.loc[unscored_mask, "suspiciousness_score"] = 1.0 - default_unscored_trust
    df.loc[unscored_mask, "risk_level"] = default_unscored_risk
    df.loc[unscored_mask, "trust_category"] = "Insufficient Evidence"
    df.loc[unscored_mask, "scoring_status"] = "insufficient_evidence_fallback"

    # Safety clipping
    df["trust_score"] = df["trust_score"].clip(0.0, 1.0)
    df["suspiciousness_score"] = df["suspiciousness_score"].clip(0.0, 1.0)
    df["purchase_probability"] = df["purchase_probability"].clip(0.0, 1.0)

    print("\nIntegration Summary:")
    print(f"Total integrated candidates: {len(df):,}")
    print(f"Unique sessions: {df['user_session'].nunique():,}")
    print(f"Unique products: {df['product_id'].nunique():,}")
    print(f"Unique brands: {df['brand'].nunique():,}")
    print("\nRisk level distribution across candidates:")
    print(df["risk_level"].value_counts())
    print("\nTrust category distribution across candidates:")
    print(df["trust_category"].value_counts())

    return df
