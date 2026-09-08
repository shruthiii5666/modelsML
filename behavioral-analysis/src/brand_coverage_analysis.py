# ============================================================
# brand_coverage_analysis.py
# PHASE 9.1 - BRAND EVIDENCE COVERAGE ANALYSIS
# ============================================================

import os
import pandas as pd

from src.config import FINAL_DATA_PATH, ANOMALY_RESULTS_PATH


def load_data():

    print("\n" + "=" * 70)
    print("PHASE 9.1")
    print("BRAND EVIDENCE COVERAGE ANALYSIS")
    print("=" * 70)

    final_df = pd.read_csv(FINAL_DATA_PATH)

    anomaly_df = pd.read_csv(
        ANOMALY_RESULTS_PATH
    )

    print(
        f"\nFinal dataset: {final_df.shape}"
    )

    print(
        f"Anomaly dataset: {anomaly_df.shape}"
    )

    return final_df, anomaly_df


def prepare_data(final_df, anomaly_df):

    # Standardize
    for df in [final_df, anomaly_df]:

        df["brand"] = (
            df["brand"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["category_code"] = (
            df["category_code"]
            .astype(str)
            .str.strip()
        )

    # Remove unknown brands
    final_df = final_df[
        final_df["brand"] != "unknown"
    ].copy()

    anomaly_df = anomaly_df[
        anomaly_df["brand"] != "unknown"
    ].copy()

    # Remove unknown category
    known_final = final_df[
        final_df["category_code"] != "unknown"
    ].copy()

    known_anomaly = anomaly_df[
        anomaly_df["category_code"] != "unknown"
    ].copy()

    return known_final, known_anomaly


def calculate_coverage(
    final_df,
    anomaly_df
):

    # ========================================================
    # ALL KNOWN CATEGORY ACTIVITY
    # ========================================================

    all_brand_categories = (
        final_df
        .groupby("brand")["category_code"]
        .nunique()
        .rename("all_known_categories")
    )

    # ========================================================
    # SUFFICIENT EVIDENCE CATEGORIES
    # ========================================================

    sufficient_df = final_df[
        final_df["sufficient_evidence"] == 1
    ].copy()

    sufficient_categories = (
        sufficient_df
        .groupby("brand")["category_code"]
        .nunique()
        .rename("sufficient_categories")
    )

    # ========================================================
    # CATEGORIES ENTERING ISOLATION FOREST
    # ========================================================

    anomaly_categories = (
        anomaly_df
        .groupby("brand")["category_code"]
        .nunique()
        .rename("anomaly_categories")
    )

    # ========================================================
    # COMBINE
    # ========================================================

    coverage = pd.concat(
        [
            all_brand_categories,
            sufficient_categories,
            anomaly_categories
        ],
        axis=1
    ).fillna(0)

    coverage[
        [
            "all_known_categories",
            "sufficient_categories",
            "anomaly_categories"
        ]
    ] = coverage[
        [
            "all_known_categories",
            "sufficient_categories",
            "anomaly_categories"
        ]
    ].astype(int)

    # ========================================================
    # COVERAGE RATE
    # ========================================================

    coverage["model_category_coverage"] = (
        coverage["anomaly_categories"]
        /
        coverage["all_known_categories"]
    )

    coverage["evidence_category_coverage"] = (
        coverage["sufficient_categories"]
        /
        coverage["all_known_categories"]
    )

    coverage = coverage.reset_index()

    return coverage


def analyze_coverage(coverage):

    print("\n" + "=" * 70)
    print("COVERAGE SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal identifiable brands: "
        f"{len(coverage):,}"
    )

    print(
        "\nDistribution of known categories per brand:"
    )

    print(
        coverage[
            "all_known_categories"
        ].describe()
    )

    print(
        "\nDistribution of sufficient categories per brand:"
    )

    print(
        coverage[
            "sufficient_categories"
        ].describe()
    )

    print(
        "\nDistribution of anomaly-model categories per brand:"
    )

    print(
        coverage[
            "anomaly_categories"
        ].describe()
    )

    # ========================================================
    # BRANDS WITH MULTIPLE CATEGORIES
    # ========================================================

    multi_category = coverage[
        coverage["all_known_categories"] > 1
    ]

    print(
        "\nBrands with >1 known category:",
        len(multi_category)
    )

    # ========================================================
    # BRANDS WITH MULTIPLE SUFFICIENT CATEGORIES
    # ========================================================

    multi_sufficient = coverage[
        coverage["sufficient_categories"] > 1
    ]

    print(
        "Brands with >1 sufficient category:",
        len(multi_sufficient)
    )

    # ========================================================
    # BRANDS WITH MULTIPLE ANOMALY OBSERVATIONS
    # ========================================================

    multi_anomaly = coverage[
        coverage["anomaly_categories"] > 1
    ]

    print(
        "Brands with >1 anomaly-model category:",
        len(multi_anomaly)
    )

    # ========================================================
    # TOP BRANDS BY CATEGORY COVERAGE
    # ========================================================

    print(
        "\nTop brands by known-category count:"
    )

    print(
        coverage
        .sort_values(
            "all_known_categories",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )

    # ========================================================
    # TOP BRANDS WITH MULTIPLE SUFFICIENT CATEGORIES
    # ========================================================

    if len(multi_sufficient) > 0:

        print(
            "\nBrands with strongest sufficient-category coverage:"
        )

        print(
            multi_sufficient
            .sort_values(
                [
                    "sufficient_categories",
                    "all_known_categories"
                ],
                ascending=False
            )
            .head(20)
            .to_string(index=False)
        )


def save_coverage(coverage):

    output_path = (
        "data/processed/"
        "brand_coverage_analysis.csv"
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    coverage.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nCoverage analysis saved:"
        f"\n{output_path}"
    )


def main():

    final_df, anomaly_df = load_data()

    final_df, anomaly_df = prepare_data(
        final_df,
        anomaly_df
    )

    coverage = calculate_coverage(
        final_df,
        anomaly_df
    )

    analyze_coverage(
        coverage
    )

    save_coverage(
        coverage
    )

    print("\n" + "=" * 70)
    print("PHASE 9.1 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()