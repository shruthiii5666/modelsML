# ============================================================
# scoring.py
# PHASE 9 - BRAND-LEVEL SUSPICIOUSNESS AGGREGATION
# ============================================================

import os
import numpy as np
import pandas as pd

from src.config import (
    FINAL_DATA_PATH,
    BRAND_SCORES_PATH,
    ANOMALY_RESULTS_PATH,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD
)

OUTPUT_COLUMNS = [
    "brand",
    "analyzed_categories",
    "anomalous_categories",
    "anomaly_category_ratio",
    "mean_anomaly_score",
    "max_anomaly_score",
    "evidence_weighted_anomaly_score",
    "suspiciousness_score",
    "risk_level",
    "scoring_status"
]


# ============================================================
# LOAD INPUT DATA
# ============================================================

def load_phase9_inputs():

    print("\n" + "=" * 70)
    print("PHASE 9")
    print("BRAND-LEVEL SUSPICIOUSNESS AGGREGATION")
    print("=" * 70)

    if not os.path.exists(ANOMALY_RESULTS_PATH):
        raise FileNotFoundError(
            f"\nAnomaly results not found:\n"
            f"{ANOMALY_RESULTS_PATH}\n\n"
            f"Run Phase 8 first."
        )

    if not os.path.exists(FINAL_DATA_PATH):
        raise FileNotFoundError(
            f"\nFinal brand dataset not found:\n"
            f"{FINAL_DATA_PATH}\n\n"
            f"Run Phase 6 first."
        )

    anomaly_df = pd.read_csv(ANOMALY_RESULTS_PATH)

    final_df = pd.read_csv(FINAL_DATA_PATH)

    print("\nInput files loaded successfully.")

    print(
        f"Anomaly results shape: {anomaly_df.shape}"
    )

    print(
        f"Final brand dataset shape: {final_df.shape}"
    )

    return anomaly_df, final_df


# ============================================================
# VALIDATE INPUT COLUMNS
# ============================================================

def validate_inputs(anomaly_df, final_df):

    anomaly_required = [
        "brand",
        "category_code",
        "is_anomaly",
        "anomaly_score"
    ]

    final_required = [
        "brand",
        "category_code",
        "total_views",
        "unique_sessions",
        "unique_users",
        "sufficient_evidence"
    ]

    missing_anomaly = [
        col for col in anomaly_required
        if col not in anomaly_df.columns
    ]

    missing_final = [
        col for col in final_required
        if col not in final_df.columns
    ]

    if missing_anomaly:
        raise ValueError(
            f"\nMissing columns in anomaly results:\n"
            f"{missing_anomaly}"
        )

    if missing_final:
        raise ValueError(
            f"\nMissing columns in final dataset:\n"
            f"{missing_final}"
        )

    print("\nInput column validation: PASSED")


# ============================================================
# PREPARE ANOMALY DATA
# ============================================================

def prepare_anomaly_data(anomaly_df, final_df):

    anomaly_df = anomaly_df.copy()
    final_df = final_df.copy()

    # --------------------------------------------------------
    # Standardize identifiers
    # --------------------------------------------------------

    anomaly_df["brand"] = (
        anomaly_df["brand"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    anomaly_df["category_code"] = (
        anomaly_df["category_code"]
        .astype(str)
        .str.strip()
    )

    final_df["brand"] = (
        final_df["brand"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    final_df["category_code"] = (
        final_df["category_code"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove unknown brand
    # --------------------------------------------------------

    anomaly_df = anomaly_df[
        anomaly_df["brand"].ne("unknown")
    ].copy()

    final_df = final_df[
        final_df["brand"].ne("unknown")
    ].copy()

    # --------------------------------------------------------
    # Keep only sufficient evidence
    #
    # Phase 8 was already trained on sufficient evidence,
    # but this ensures Phase 9 remains protected.
    # --------------------------------------------------------

    final_df = final_df[
        final_df["sufficient_evidence"] == 1
    ].copy()

    # --------------------------------------------------------
    # Remove unknown category from final brand scoring
    #
    # Category-relative comparison is not meaningful when
    # category information is unknown.
    # --------------------------------------------------------

    anomaly_df = anomaly_df[
        anomaly_df["category_code"].ne("unknown")
    ].copy()

    final_df = final_df[
        final_df["category_code"].ne("unknown")
    ].copy()

    print("\nAfter filtering:")

    print(
        f"Anomaly observations: {len(anomaly_df):,}"
    )

    print(
        f"Known-category evidence rows: {len(final_df):,}"
    )

    print(
        f"Brands in anomaly data: "
        f"{anomaly_df['brand'].nunique():,}"
    )

    print(
        f"Known categories: "
        f"{final_df['category_code'].nunique():,}"
    )

    return anomaly_df, final_df


# ============================================================
# MERGE ANOMALY + EVIDENCE
# ============================================================

def merge_anomaly_with_evidence(anomaly_df, final_df):

    evidence_columns = [
        "brand",
        "category_code",
        "total_views",
        "unique_sessions",
        "unique_users",
        "total_events",
        "sufficient_evidence"
    ]

    evidence_df = final_df[evidence_columns].copy()

    # --------------------------------------------------------
    # Remove duplicate brand-category combinations
    # --------------------------------------------------------

    evidence_df = evidence_df.drop_duplicates(
        subset=["brand", "category_code"]
    )

    merged = anomaly_df.merge(
        evidence_df,
        on=["brand", "category_code"],
        how="inner"
    )

    print("\nAnomaly + evidence merge completed.")

    print(
        f"Merged observations: {len(merged):,}"
    )

    if len(merged) == 0:
        raise ValueError(
            "\nNo matching brand-category observations "
            "were found between Phase 8 and Phase 6."
        )

    # --------------------------------------------------------
    # Validate anomaly score
    # --------------------------------------------------------

    merged["anomaly_score"] = pd.to_numeric(
        merged["anomaly_score"],
        errors="coerce"
    )

    merged["is_anomaly"] = pd.to_numeric(
        merged["is_anomaly"],
        errors="coerce"
    )

    merged["total_views"] = pd.to_numeric(
        merged["total_views"],
        errors="coerce"
    )

    merged["unique_sessions"] = pd.to_numeric(
        merged["unique_sessions"],
        errors="coerce"
    )

    merged["unique_users"] = pd.to_numeric(
        merged["unique_users"],
        errors="coerce"
    )

    merged["total_events"] = pd.to_numeric(
        merged["total_events"],
        errors="coerce"
    )

    merged = merged.dropna(
        subset=[
            "anomaly_score",
            "total_views",
            "unique_sessions",
            "unique_users"
        ]
    )

    return merged


# ============================================================
# CALCULATE EVIDENCE WEIGHT
# ============================================================

def calculate_evidence_weight(df):

    df = df.copy()

    # --------------------------------------------------------
    # Log transformation prevents high-volume brands from
    # completely dominating the score.
    # --------------------------------------------------------

    log_views = np.log1p(
        df["total_views"].clip(lower=0)
    )

    log_sessions = np.log1p(
        df["unique_sessions"].clip(lower=0)
    )

    log_users = np.log1p(
        df["unique_users"].clip(lower=0)
    )

    # --------------------------------------------------------
    # Equal contribution from views, sessions and users
    # --------------------------------------------------------

    df["evidence_weight"] = (
        log_views +
        log_sessions +
        log_users
    ) / 3.0

    # Prevent zero weights
    df["evidence_weight"] = (
        df["evidence_weight"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df


# ============================================================
# AGGREGATE BRAND-LEVEL SIGNALS
# ============================================================

def aggregate_brand_scores(df):

    results = []

    # --------------------------------------------------------
    # Process each brand
    # --------------------------------------------------------

    for brand, group in df.groupby("brand"):

        group = group.copy()

        # ----------------------------------------------------
        # Number of analyzed categories
        # ----------------------------------------------------

        analyzed_categories = (
            group["category_code"]
            .nunique()
        )

        # ----------------------------------------------------
        # Number of anomalous categories
        # ----------------------------------------------------

        anomalous_categories = int(
            group["is_anomaly"].sum()
        )

        # ----------------------------------------------------
        # Anomaly category ratio
        # ----------------------------------------------------

        if analyzed_categories > 0:

            anomaly_category_ratio = (
                anomalous_categories /
                analyzed_categories
            )

        else:

            anomaly_category_ratio = 0.0

        # ----------------------------------------------------
        # Mean anomaly score
        # ----------------------------------------------------

        mean_anomaly_score = (
            group["anomaly_score"]
            .mean()
        )

        # ----------------------------------------------------
        # Maximum anomaly score
        # ----------------------------------------------------

        max_anomaly_score = (
            group["anomaly_score"]
            .max()
        )

        # ----------------------------------------------------
        # Evidence-weighted anomaly score
        # ----------------------------------------------------

        weights = group["evidence_weight"]

        scores = group["anomaly_score"]

        total_weight = weights.sum()

        if total_weight > 0:

            evidence_weighted_score = (
                (scores * weights).sum()
                / total_weight
            )

        else:

            evidence_weighted_score = (
                scores.mean()
            )

        # ----------------------------------------------------
        # Main suspiciousness score
        #
        # Evidence-weighted anomaly is the primary signal.
        #
        # We combine it with anomaly frequency so that a
        # brand repeatedly showing anomalous behavior across
        # categories receives a stronger score.
        # ----------------------------------------------------

        suspiciousness_score = (
            0.80 * evidence_weighted_score
            +
            0.20 * anomaly_category_ratio
        )

        # ----------------------------------------------------
        # Clamp to [0,1]
        # ----------------------------------------------------

        suspiciousness_score = float(
            np.clip(
                suspiciousness_score,
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if suspiciousness_score < LOW_RISK_THRESHOLD:

            risk_level = "Low"

        elif suspiciousness_score < HIGH_RISK_THRESHOLD:

            risk_level = "Medium"

        else:

            risk_level = "High"

        results.append({
            "brand": brand,
            "analyzed_categories": analyzed_categories,
            "anomalous_categories": anomalous_categories,
            "anomaly_category_ratio": anomaly_category_ratio,
            "mean_anomaly_score": mean_anomaly_score,
            "max_anomaly_score": max_anomaly_score,
            "evidence_weighted_anomaly_score":
                evidence_weighted_score,
            "suspiciousness_score":
                suspiciousness_score,
            "risk_level": risk_level,
            "scoring_status": "scored"
        })

    return pd.DataFrame(results)


# ============================================================
# ADD BRANDS WITHOUT KNOWN CATEGORY EVIDENCE
# ============================================================

def add_unscored_brands(
    scores_df,
    final_df
):

    # --------------------------------------------------------
    # All identifiable brands in the final dataset
    # --------------------------------------------------------

    all_brands = set(
        final_df["brand"]
        .dropna()
        .unique()
    )

    scored_brands = set(
        scores_df["brand"]
        .dropna()
        .unique()
    )

    unscored_brands = (
        all_brands - scored_brands
    )

    if not unscored_brands:

        return scores_df

    rows = []

    for brand in sorted(unscored_brands):

        rows.append({
            "brand": brand,
            "analyzed_categories": 0,
            "anomalous_categories": 0,
            "anomaly_category_ratio": np.nan,
            "mean_anomaly_score": np.nan,
            "max_anomaly_score": np.nan,
            "evidence_weighted_anomaly_score": np.nan,
            "suspiciousness_score": np.nan,
            "risk_level": "Not Scored",
            "scoring_status":
                "no_known_category_evidence"
        })

    additional_df = pd.DataFrame(rows)

    scores_df = pd.concat(
        [scores_df, additional_df],
        ignore_index=True
    )

    return scores_df


# ============================================================
# SAVE RESULTS
# ============================================================

def save_brand_scores(scores_df):

    output_directory = os.path.dirname(
        BRAND_SCORES_PATH
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    scores_df = scores_df.sort_values(
        by="suspiciousness_score",
        ascending=False,
        na_position="last"
    )

    scores_df.to_csv(
        BRAND_SCORES_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("BRAND SUSPICIOUSNESS DATASET SAVED")
    print("=" * 70)

    print(
        f"\nPath: {BRAND_SCORES_PATH}"
    )

    print(
        f"Shape: {scores_df.shape}"
    )

    return scores_df


# ============================================================
# ANALYSIS
# ============================================================

def analyze_brand_scores(scores_df):

    scored = scores_df[
        scores_df["scoring_status"] == "scored"
    ].copy()

    print("\n" + "=" * 70)
    print("PHASE 9 BRAND SCORE ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal brands in output: "
        f"{len(scores_df):,}"
    )

    print(
        f"Brands with numeric score: "
        f"{len(scored):,}"
    )

    print(
        f"Brands not scored: "
        f"{len(scores_df) - len(scored):,}"
    )

    if len(scored) == 0:

        print(
            "\nNo brands received a numeric score."
        )

        return

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print("\nRisk-level distribution:")

    print(
        scored["risk_level"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Suspiciousness statistics
    # --------------------------------------------------------

    print("\nSuspiciousness statistics:")

    print(
        scored["suspiciousness_score"]
        .describe()
    )

    # --------------------------------------------------------
    # Top suspicious brands
    # --------------------------------------------------------

    print(
        "\nTop 20 brands by suspiciousness:"
    )

    display_columns = [
        "brand",
        "analyzed_categories",
        "anomalous_categories",
        "anomaly_category_ratio",
        "mean_anomaly_score",
        "max_anomaly_score",
        "evidence_weighted_anomaly_score",
        "suspiciousness_score",
        "risk_level"
    ]

    print(
        scored[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Lowest suspiciousness brands
    # --------------------------------------------------------

    print(
        "\n20 brands with lowest suspiciousness:"
    )

    print(
        scored[
            display_columns
        ]
        .sort_values(
            "suspiciousness_score",
            ascending=True
        )
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_phase9():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    anomaly_df, final_df = (
        load_phase9_inputs()
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_inputs(
        anomaly_df,
        final_df
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    anomaly_df, final_df = (
        prepare_anomaly_data(
            anomaly_df,
            final_df
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged_df = (
        merge_anomaly_with_evidence(
            anomaly_df,
            final_df
        )
    )

    # --------------------------------------------------------
    # Evidence weighting
    # --------------------------------------------------------

    merged_df = (
        calculate_evidence_weight(
            merged_df
        )
    )

    # --------------------------------------------------------
    # Aggregate by brand
    # --------------------------------------------------------

    scores_df = (
        aggregate_brand_scores(
            merged_df
        )
    )

    # --------------------------------------------------------
    # Add brands that cannot be scored
    # --------------------------------------------------------

    scores_df = (
        add_unscored_brands(
            scores_df,
            final_df
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    scores_df = save_brand_scores(
        scores_df
    )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    analyze_brand_scores(
        scores_df
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 9 FINAL CHECK")
    print("=" * 70)

    print(
        f"Output shape: {scores_df.shape}"
    )

    print(
        "Missing brand names:",
        scores_df["brand"].isna().sum()
    )

    print(
        "Duplicate brands:",
        scores_df["brand"].duplicated().sum()
    )

    scored = scores_df[
        scores_df["scoring_status"] == "scored"
    ]

    print(
        "Scored brands:",
        len(scored)
    )

    if len(scored) > 0:

        print(
            "Suspiciousness minimum:",
            scored["suspiciousness_score"].min()
        )

        print(
            "Suspiciousness maximum:",
            scored["suspiciousness_score"].max()
        )

        print(
            "Suspiciousness mean:",
            scored["suspiciousness_score"].mean()
        )

    print(
        "\nPHASE 9 COMPLETED SUCCESSFULLY"
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    run_phase9()