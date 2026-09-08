# ============================================================
# category_relative_features.py
# Phase 6 - Category-Relative Behavioral Feature Extraction
# ============================================================

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    CHUNK_SIZE,
    BRAND_CATEGORY_DATA_PATH,
    FINAL_DATA_PATH
)

from src.data_loader import read_preprocessed_chunks


# ============================================================
# 1. LOAD ONLY FIRST 3 CHUNKS
# ============================================================

def load_first_n_cleaned_chunks(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE,
    max_chunks=3
):
    """
    Read and clean only the first max_chunks chunks.

    IMPORTANT:
    This function NEVER processes the complete dataset.

    With:
        chunk_size = 500,000
        max_chunks = 3

    maximum rows read = 1,500,000
    """

    cleaned_chunks = []

    print("\n")
    print("=" * 60)
    print("LOADING DATA FOR PHASE 6")
    print("=" * 60)

    print("\nChunk size:", f"{chunk_size:,}")
    print("Maximum chunks:", max_chunks)
    print(
        "Maximum raw rows:",
        f"{chunk_size * max_chunks:,}"
    )

    for chunk_number, df in enumerate(
        read_preprocessed_chunks(
            file_path,
            chunk_size
        ),
        start=1
    ):

        print(
            f"\nChunk {chunk_number}: "
            f"{len(df):,} cleaned rows"
        )

        cleaned_chunks.append(df)

        # ----------------------------------------------------
        # STOP AFTER FIRST 3 CHUNKS
        # ----------------------------------------------------

        if chunk_number >= max_chunks:

            print(
                "\nReached the 3-chunk limit."
            )

            break

    if not cleaned_chunks:
        raise ValueError(
            "No cleaned data was loaded."
        )

    combined = pd.concat(
        cleaned_chunks,
        ignore_index=True
    )

    print("\n")
    print("=" * 60)
    print("PHASE 6 INPUT DATA")
    print("=" * 60)

    print(
        "\nCombined cleaned rows:",
        f"{len(combined):,}"
    )

    return combined


# ============================================================
# 2. CREATE EXACT BRAND × CATEGORY FEATURES
# ============================================================

def create_exact_brand_category_features(df):

    print("\n")
    print("=" * 60)
    print("CREATING EXACT BRAND × CATEGORY FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Remove unknown brands
    # --------------------------------------------------------

    df = df[
        df["brand"] != "unknown"
    ].copy()

    # --------------------------------------------------------
    # Keep unknown category for now
    #
    # We DO NOT remove unknown category because it represents
    # a significant portion of the available behavioral data.
    # --------------------------------------------------------

    df["category_code"] = (
        df["category_code"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    df.loc[
        df["category_code"] == "",
        "category_code"
    ] = "unknown"

    # --------------------------------------------------------
    # Global aggregation
    #
    # Because all 3 chunks are combined before groupby,
    # nunique() now calculates exact distinct values across
    # the complete 1.5M-row working subset.
    # --------------------------------------------------------

    result = (
        df.groupby(
            [
                "brand",
                "category_code"
            ]
        )
        .agg(
            total_events=(
                "event_type",
                "size"
            ),

            total_views=(
                "is_view",
                "sum"
            ),

            total_carts=(
                "is_cart",
                "sum"
            ),

            total_purchases=(
                "is_purchase",
                "sum"
            ),

            unique_sessions=(
                "user_session",
                "nunique"
            ),

            unique_users=(
                "user_id",
                "nunique"
            ),

            unique_products=(
                "product_id",
                "nunique"
            ),

            unique_categories=(
                "category_code",
                "nunique"
            ),

            average_price=(
                "price",
                "mean"
            ),

            total_revenue=(
                "price",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Behavioral rates
    # --------------------------------------------------------

    result["view_to_cart_rate"] = np.where(
        result["total_views"] > 0,
        result["total_carts"]
        / result["total_views"],
        0
    )

    result["cart_to_purchase_rate"] = np.where(
        result["total_carts"] > 0,
        result["total_purchases"]
        / result["total_carts"],
        0
    )

    result["view_to_purchase_rate"] = np.where(
        result["total_views"] > 0,
        result["total_purchases"]
        / result["total_views"],
        0
    )

    result["purchase_event_rate"] = np.where(
        result["total_events"] > 0,
        result["total_purchases"]
        / result["total_events"],
        0
    )

    result["cart_event_rate"] = np.where(
        result["total_events"] > 0,
        result["total_carts"]
        / result["total_events"],
        0
    )

    result["view_event_rate"] = np.where(
        result["total_events"] > 0,
        result["total_views"]
        / result["total_events"],
        0
    )

    return result


# ============================================================
# 3. CREATE CATEGORY-LEVEL BASELINES
# ============================================================

def create_category_baselines(
    brand_category_df
):

    print("\n")
    print("=" * 60)
    print("CREATING CATEGORY BASELINES")
    print("=" * 60)

    # --------------------------------------------------------
    # Category-level totals
    # --------------------------------------------------------

    category_totals = (
        brand_category_df
        .groupby("category_code")
        .agg(
            category_total_events=(
                "total_events",
                "sum"
            ),

            category_total_views=(
                "total_views",
                "sum"
            ),

            category_total_carts=(
                "total_carts",
                "sum"
            ),

            category_total_purchases=(
                "total_purchases",
                "sum"
            ),

            category_average_price=(
                "average_price",
                "mean"
            ),

            category_brands=(
                "brand",
                "nunique"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Category conversion rates
    # --------------------------------------------------------

    category_totals[
        "category_view_to_cart_rate"
    ] = np.where(
        category_totals[
            "category_total_views"
        ] > 0,

        category_totals[
            "category_total_carts"
        ]
        /
        category_totals[
            "category_total_views"
        ],

        0
    )

    category_totals[
        "category_cart_to_purchase_rate"
    ] = np.where(
        category_totals[
            "category_total_carts"
        ] > 0,

        category_totals[
            "category_total_purchases"
        ]
        /
        category_totals[
            "category_total_carts"
        ],

        0
    )

    category_totals[
        "category_view_to_purchase_rate"
    ] = np.where(
        category_totals[
            "category_total_views"
        ] > 0,

        category_totals[
            "category_total_purchases"
        ]
        /
        category_totals[
            "category_total_views"
        ],

        0
    )

    category_totals[
        "category_purchase_event_rate"
    ] = np.where(
        category_totals[
            "category_total_events"
        ] > 0,

        category_totals[
            "category_total_purchases"
        ]
        /
        category_totals[
            "category_total_events"
        ],

        0
    )

    category_totals[
        "category_cart_event_rate"
    ] = np.where(
        category_totals[
            "category_total_events"
        ] > 0,

        category_totals[
            "category_total_carts"
        ]
        /
        category_totals[
            "category_total_events"
        ],

        0
    )

    category_totals[
        "category_view_event_rate"
    ] = np.where(
        category_totals[
            "category_total_events"
        ] > 0,

        category_totals[
            "category_total_views"
        ]
        /
        category_totals[
            "category_total_events"
        ],

        0
    )

    return category_totals


# ============================================================
# 4. MERGE CATEGORY BASELINES
# ============================================================

def merge_category_baselines(
    brand_category_df,
    category_baselines
):

    result = brand_category_df.merge(
        category_baselines,
        on="category_code",
        how="left"
    )

    return result


# ============================================================
# 5. CREATE RELATIVE DEVIATIONS
# ============================================================

def create_relative_deviation_features(
    df
):

    print("\n")
    print("=" * 60)
    print("CREATING CATEGORY-RELATIVE DEVIATION FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Conversion-rate deviations
    # --------------------------------------------------------

    df[
        "view_to_cart_deviation"
    ] = (
        df["view_to_cart_rate"]
        -
        df["category_view_to_cart_rate"]
    )

    df[
        "cart_to_purchase_deviation"
    ] = (
        df["cart_to_purchase_rate"]
        -
        df["category_cart_to_purchase_rate"]
    )

    df[
        "view_to_purchase_deviation"
    ] = (
        df["view_to_purchase_rate"]
        -
        df["category_view_to_purchase_rate"]
    )

    df[
        "purchase_event_deviation"
    ] = (
        df["purchase_event_rate"]
        -
        df["category_purchase_event_rate"]
    )

    df[
        "cart_event_deviation"
    ] = (
        df["cart_event_rate"]
        -
        df["category_cart_event_rate"]
    )

    df[
        "view_event_deviation"
    ] = (
        df["view_event_rate"]
        -
        df["category_view_event_rate"]
    )

    # --------------------------------------------------------
    # Price deviation
    # --------------------------------------------------------

    df[
        "price_deviation"
    ] = (
        df["average_price"]
        -
        df["category_average_price"]
    )

    # --------------------------------------------------------
    # Relative price ratio
    #
    # Example:
    # 1.0 = same as category average
    # 2.0 = twice category average
    # 0.5 = half category average
    # --------------------------------------------------------

    df[
        "price_relative_ratio"
    ] = np.where(
        df["category_average_price"] > 0,

        df["average_price"]
        /
        df["category_average_price"],

        0
    )

    # --------------------------------------------------------
    # Category event shares
    # --------------------------------------------------------

    df[
        "event_share_in_category"
    ] = np.where(
        df["category_total_events"] > 0,

        df["total_events"]
        /
        df["category_total_events"],

        0
    )

    df[
        "view_share_in_category"
    ] = np.where(
        df["category_total_views"] > 0,

        df["total_views"]
        /
        df["category_total_views"],

        0
    )

    df[
        "cart_share_in_category"
    ] = np.where(
        df["category_total_carts"] > 0,

        df["total_carts"]
        /
        df["category_total_carts"],

        0
    )

    df[
        "purchase_share_in_category"
    ] = np.where(
        df["category_total_purchases"] > 0,

        df["total_purchases"]
        /
        df["category_total_purchases"],

        0
    )

    # --------------------------------------------------------
    # Session/user/product normalized features
    # --------------------------------------------------------

    df[
        "events_per_session"
    ] = np.where(
        df["unique_sessions"] > 0,

        df["total_events"]
        /
        df["unique_sessions"],

        0
    )

    df[
        "views_per_session"
    ] = np.where(
        df["unique_sessions"] > 0,

        df["total_views"]
        /
        df["unique_sessions"],

        0
    )

    df[
        "purchases_per_session"
    ] = np.where(
        df["unique_sessions"] > 0,

        df["total_purchases"]
        /
        df["unique_sessions"],

        0
    )

    df[
        "events_per_user"
    ] = np.where(
        df["unique_users"] > 0,

        df["total_events"]
        /
        df["unique_users"],

        0
    )

    df[
        "purchases_per_user"
    ] = np.where(
        df["unique_users"] > 0,

        df["total_purchases"]
        /
        df["unique_users"],

        0
    )

    df[
        "events_per_product"
    ] = np.where(
        df["unique_products"] > 0,

        df["total_events"]
        /
        df["unique_products"],

        0
    )

    return df


# ============================================================
# 6. ADD MINIMUM EVIDENCE INDICATOR
# ============================================================

def create_evidence_features(df):

    print("\n")
    print("=" * 60)
    print("CREATING EVIDENCE FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Evidence conditions
    # --------------------------------------------------------

    df["sufficient_views"] = (
        df["total_views"] >= 20
    ).astype("int8")

    df["sufficient_sessions"] = (
        df["unique_sessions"] >= 10
    ).astype("int8")

    df["sufficient_users"] = (
        df["unique_users"] >= 5
    ).astype("int8")

    # --------------------------------------------------------
    # Overall evidence flag
    #
    # All three conditions must be satisfied.
    # --------------------------------------------------------

    df["sufficient_evidence"] = (
        (
            df["sufficient_views"] == 1
        )
        &
        (
            df["sufficient_sessions"] == 1
        )
        &
        (
            df["sufficient_users"] == 1
        )
    ).astype("int8")

    return df


# ============================================================
# 7. SAVE FINAL DATASET
# ============================================================

def save_final_dataset(
    df,
    output_path=FINAL_DATA_PATH
):

    df.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print("=" * 60)
    print("FINAL DATASET SAVED")
    print("=" * 60)

    print("\nPath:")
    print(output_path)

    print("\nShape:")
    print(df.shape)


# ============================================================
# 8. MAIN PHASE 6 FUNCTION
# ============================================================

def create_category_relative_features(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE,
    max_chunks=3
):

    print("\n")
    print("=" * 60)
    print("PHASE 6 - CATEGORY-RELATIVE FEATURE EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load ONLY first 3 chunks
    # --------------------------------------------------------

    df = load_first_n_cleaned_chunks(
        file_path=file_path,
        chunk_size=chunk_size,
        max_chunks=max_chunks
    )

    # --------------------------------------------------------
    # Exact brand × category aggregation
    # --------------------------------------------------------

    brand_category_df = (
        create_exact_brand_category_features(
            df
        )
    )

    # --------------------------------------------------------
    # Save corrected brand × category dataset
    # --------------------------------------------------------

    brand_category_df.to_csv(
        BRAND_CATEGORY_DATA_PATH,
        index=False
    )

    print(
        "\nCorrected brand × category dataset saved:"
    )

    print(
        BRAND_CATEGORY_DATA_PATH
    )

    # --------------------------------------------------------
    # Category baselines
    # --------------------------------------------------------

    category_baselines = (
        create_category_baselines(
            brand_category_df
        )
    )

    # --------------------------------------------------------
    # Merge baselines
    # --------------------------------------------------------

    final_df = merge_category_baselines(
        brand_category_df,
        category_baselines
    )

    # --------------------------------------------------------
    # Relative deviations
    # --------------------------------------------------------

    final_df = (
        create_relative_deviation_features(
            final_df
        )
    )

    # --------------------------------------------------------
    # Evidence features
    # --------------------------------------------------------

    final_df = (
        create_evidence_features(
            final_df
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        [
            "category_code",
            "total_events"
        ],
        ascending=[
            True,
            False
        ]
    )

    final_df = final_df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save final dataset
    # --------------------------------------------------------

    save_final_dataset(
        final_df,
        FINAL_DATA_PATH
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PHASE 6 COMPLETED")
    print("=" * 60)

    print(
        "\nBrand × Category combinations:",
        f"{len(brand_category_df):,}"
    )

    print(
        "\nCategories:",
        f"{brand_category_df['category_code'].nunique():,}"
    )

    print(
        "\nBrands:",
        f"{brand_category_df['brand'].nunique():,}"
    )

    print(
        "\nSufficient-evidence combinations:",
        int(
            final_df[
                "sufficient_evidence"
            ].sum()
        )
    )

    print(
        "\nFinal dataset:",
        FINAL_DATA_PATH
    )

    return final_df