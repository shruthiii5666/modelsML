# ============================================================
# trust_score.py
# PHASE 10 - BRAND TRUST SCORE CALCULATION
# ============================================================

import os
import numpy as np
import pandas as pd

from src.config import (
    BRAND_SCORES_PATH,
    TRUST_DATASET_PATH,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD
)


# ============================================================
# LOAD BRAND SCORES
# ============================================================

def load_brand_scores():

    print("\n" + "=" * 70)
    print("PHASE 10")
    print("BRAND TRUST SCORE CALCULATION")
    print("=" * 70)

    if not os.path.exists(BRAND_SCORES_PATH):

        raise FileNotFoundError(
            f"\nBrand score file not found:\n"
            f"{BRAND_SCORES_PATH}\n\n"
            f"Run Phase 9 first."
        )

    df = pd.read_csv(
        BRAND_SCORES_PATH
    )

    print("\nBrand score dataset loaded.")

    print(
        f"Input path: {BRAND_SCORES_PATH}"
    )

    print(
        f"Input shape: {df.shape}"
    )

    return df


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df):

    required_columns = [
        "brand",
        "suspiciousness_score",
        "risk_level",
        "scoring_status"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            f"{missing_columns}"
        )

    print(
        "\nInput validation: PASSED"
    )


# ============================================================
# CALCULATE TRUST SCORE
# ============================================================

def calculate_trust_score(df):

    df = df.copy()

    # --------------------------------------------------------
    # Convert suspiciousness to numeric
    # --------------------------------------------------------

    df["suspiciousness_score"] = pd.to_numeric(
        df["suspiciousness_score"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Initialize trust score
    # --------------------------------------------------------

    df["trust_score"] = np.nan

    # --------------------------------------------------------
    # Only calculate trust for brands that have a valid
    # suspiciousness score.
    # --------------------------------------------------------

    scored_mask = (
        df["scoring_status"] == "scored"
    )

    # --------------------------------------------------------
    # Trust = 1 - Suspiciousness
    # --------------------------------------------------------

    df.loc[
        scored_mask,
        "trust_score"
    ] = (
        1.0
        -
        df.loc[
            scored_mask,
            "suspiciousness_score"
        ]
    )

    # --------------------------------------------------------
    # Numerical safety
    # --------------------------------------------------------

    df["trust_score"] = np.clip(
        df["trust_score"],
        0.0,
        1.0
    )

    return df


# ============================================================
# ASSIGN TRUST CATEGORY
# ============================================================

def assign_trust_category(df):

    df = df.copy()

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    df["trust_category"] = "Insufficient Evidence"

    # --------------------------------------------------------
    # High Trust
    #
    # Trust > 0.67
    # Equivalent to suspiciousness < 0.33
    # --------------------------------------------------------

    high_trust = (
        df["trust_score"] >
        HIGH_RISK_THRESHOLD
    )

    # --------------------------------------------------------
    # Medium Trust
    #
    # 0.33 < Trust <= 0.67
    # --------------------------------------------------------

    medium_trust = (
        (df["trust_score"] >
         LOW_RISK_THRESHOLD)
        &
        (df["trust_score"] <=
         HIGH_RISK_THRESHOLD)
    )

    # --------------------------------------------------------
    # Low Trust
    #
    # Trust <= 0.33
    # --------------------------------------------------------

    low_trust = (
        df["trust_score"] <=
        LOW_RISK_THRESHOLD
    )

    df.loc[
        high_trust,
        "trust_category"
    ] = "High Trust"

    df.loc[
        medium_trust,
        "trust_category"
    ] = "Medium Trust"

    df.loc[
        low_trust,
        "trust_category"
    ] = "Low Trust"

    # --------------------------------------------------------
    # Explicitly preserve insufficient evidence status
    # --------------------------------------------------------

    insufficient = (
        df["scoring_status"]
        != "scored"
    )

    df.loc[
        insufficient,
        "trust_category"
    ] = "Insufficient Evidence"

    return df


# ============================================================
# ADD TRUST INTERPRETATION
# ============================================================

def add_trust_interpretation(df):

    df = df.copy()

    df["trust_interpretation"] = (
        "Insufficient behavioral evidence"
    )

    scored = (
        df["scoring_status"] == "scored"
    )

    df.loc[
        scored &
        (df["trust_category"] == "High Trust"),
        "trust_interpretation"
    ] = (
        "Behavior is relatively consistent "
        "with observed category patterns"
    )

    df.loc[
        scored &
        (df["trust_category"] == "Medium Trust"),
        "trust_interpretation"
    ] = (
        "Behavior shows moderate deviation "
        "from observed category patterns"
    )

    df.loc[
        scored &
        (df["trust_category"] == "Low Trust"),
        "trust_interpretation"
    ] = (
        "Behavior shows relatively high "
        "deviation from observed category patterns"
    )

    return df


# ============================================================
# SAVE TRUST DATASET
# ============================================================

def save_trust_dataset(df):

    output_directory = os.path.dirname(
        TRUST_DATASET_PATH
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Arrange columns
    # --------------------------------------------------------

    preferred_columns = [
        "brand",

        "analyzed_categories",
        "anomalous_categories",
        "anomaly_category_ratio",

        "mean_anomaly_score",
        "max_anomaly_score",
        "evidence_weighted_anomaly_score",

        "suspiciousness_score",
        "risk_level",

        "trust_score",
        "trust_category",
        "trust_interpretation",

        "scoring_status"
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in existing_columns
    ]

    df = df[
        existing_columns +
        remaining_columns
    ]

    # --------------------------------------------------------
    # Sort scored brands by trust
    # --------------------------------------------------------

    df = df.sort_values(
        by="trust_score",
        ascending=False,
        na_position="last"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        TRUST_DATASET_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("TRUST DATASET SAVED")
    print("=" * 70)

    print(
        f"\nPath: {TRUST_DATASET_PATH}"
    )

    print(
        f"Shape: {df.shape}"
    )

    return df


# ============================================================
# TRUST SCORE ANALYSIS
# ============================================================

def analyze_trust_scores(df):

    print("\n" + "=" * 70)
    print("PHASE 10 TRUST SCORE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Total brands
    # --------------------------------------------------------

    print(
        f"\nTotal brands: {len(df):,}"
    )

    # --------------------------------------------------------
    # Numeric trust scores
    # --------------------------------------------------------

    scored_df = df[
        df["scoring_status"] == "scored"
    ].copy()

    insufficient_df = df[
        df["scoring_status"] != "scored"
    ].copy()

    print(
        f"Brands with trust score: "
        f"{len(scored_df):,}"
    )

    print(
        f"Brands without trust score: "
        f"{len(insufficient_df):,}"
    )

    # --------------------------------------------------------
    # Trust statistics
    # --------------------------------------------------------

    if len(scored_df) > 0:

        print(
            "\nTrust score statistics:"
        )

        print(
            scored_df[
                "trust_score"
            ].describe()
        )

    # --------------------------------------------------------
    # Trust category distribution
    # --------------------------------------------------------

    print(
        "\nTrust category distribution:"
    )

    print(
        df[
            "trust_category"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print(
        "\nRisk-level distribution:"
    )

    print(
        df[
            "risk_level"
        ].value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Highest trust brands
    # --------------------------------------------------------

    if len(scored_df) > 0:

        print(
            "\nTop 20 highest-trust brands:"
        )

        columns = [
            "brand",
            "analyzed_categories",
            "anomalous_categories",
            "suspiciousness_score",
            "trust_score",
            "trust_category"
        ]

        print(
            scored_df[
                columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # Lowest trust brands
        # ----------------------------------------------------

        print(
            "\n20 lowest-trust brands:"
        )

        print(
            scored_df
            .sort_values(
                "trust_score",
                ascending=True
            )
            [
                columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_dataset(df):

    print("\n" + "=" * 70)
    print("PHASE 10 FINAL CHECK")
    print("=" * 70)

    print(
        f"Output shape: {df.shape}"
    )

    print(
        "Missing brand names:",
        df["brand"].isna().sum()
    )

    print(
        "Duplicate brands:",
        df["brand"].duplicated().sum()
    )

    # --------------------------------------------------------
    # Trust score range
    # --------------------------------------------------------

    valid_trust = df[
        "trust_score"
    ].dropna()

    if len(valid_trust) > 0:

        print(
            "\nTrust score minimum:",
            valid_trust.min()
        )

        print(
            "Trust score maximum:",
            valid_trust.max()
        )

        print(
            "Trust score mean:",
            valid_trust.mean()
        )

        outside_range = (
            (valid_trust < 0)
            |
            (valid_trust > 1)
        ).sum()

        print(
            "Trust scores outside [0,1]:",
            outside_range
        )

    # --------------------------------------------------------
    # Consistency check
    # --------------------------------------------------------

    scored = df[
        df["scoring_status"] == "scored"
    ].copy()

    if len(scored) > 0:

        expected_trust = (
            1.0
            -
            scored["suspiciousness_score"]
        )

        difference = (
            scored["trust_score"]
            -
            expected_trust
        ).abs().max()

        print(
            "Maximum Trust = 1 - Suspiciousness "
            "difference:",
            difference
        )

    print(
        "\nPHASE 10 COMPLETED SUCCESSFULLY"
    )


# ============================================================
# MAIN
# ============================================================

def run_phase10():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_brand_scores()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input(df)

    # --------------------------------------------------------
    # Trust score
    # --------------------------------------------------------

    df = calculate_trust_score(
        df
    )

    # --------------------------------------------------------
    # Trust category
    # --------------------------------------------------------

    df = assign_trust_category(
        df
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    df = add_trust_interpretation(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df = save_trust_dataset(
        df
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    analyze_trust_scores(
        df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_final_dataset(
        df
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_phase10()