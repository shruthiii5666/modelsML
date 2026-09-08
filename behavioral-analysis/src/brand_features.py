# ============================================================
# brand_features.py
# Phase 4 - Brand-Level Behavioral Feature Extraction
# ============================================================

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    CHUNK_SIZE,
    BRAND_DATA_PATH
)

from src.data_loader import (
    read_preprocessed_chunks
)


# ============================================================
# 1. AGGREGATE BRAND BEHAVIOR FROM ONE CHUNK
# ============================================================

def aggregate_brand_chunk(df):
    """
    Aggregate event behavior at brand level for one chunk.
    """

    brand_df = (
        df.groupby("brand")
        .agg(
            total_events=("event_type", "size"),

            total_views=("is_view", "sum"),

            total_carts=("is_cart", "sum"),

            total_purchases=("is_purchase", "sum"),

            unique_sessions=("user_session", "nunique"),

            unique_users=("user_id", "nunique"),

            unique_products=("product_id", "nunique"),

            unique_categories=("category_code", "nunique"),

            average_price=("price", "mean"),

            total_revenue=("price", "sum")
        )
        .reset_index()
    )

    return brand_df


# ============================================================
# 2. COMBINE BRAND CHUNK RESULTS
# ============================================================

def combine_brand_chunks(brand_chunks):
    """
    Combine brand-level results from multiple chunks.
    """

    if not brand_chunks:
        return pd.DataFrame()

    combined = pd.concat(
        brand_chunks,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Aggregate numerical values
    # --------------------------------------------------------

    final_brand = (
        combined
        .groupby("brand")
        .agg(
            total_events=("total_events", "sum"),

            total_views=("total_views", "sum"),

            total_carts=("total_carts", "sum"),

            total_purchases=("total_purchases", "sum"),

            unique_sessions=("unique_sessions", "sum"),

            unique_users=("unique_users", "sum"),

            unique_products=("unique_products", "sum"),

            unique_categories=("unique_categories", "sum"),

            average_price=("average_price", "mean"),

            total_revenue=("total_revenue", "sum")
        )
        .reset_index()
    )

    return final_brand


# ============================================================
# 3. CREATE BEHAVIORAL RATES
# ============================================================

def create_brand_behavior_rates(df):
    """
    Create behavioral conversion and interaction rates.
    """

    # --------------------------------------------------------
    # View -> Cart
    # --------------------------------------------------------

    df["view_to_cart_rate"] = np.where(
        df["total_views"] > 0,
        df["total_carts"] / df["total_views"],
        0
    )

    # --------------------------------------------------------
    # Cart -> Purchase
    # --------------------------------------------------------

    df["cart_to_purchase_rate"] = np.where(
        df["total_carts"] > 0,
        df["total_purchases"] / df["total_carts"],
        0
    )

    # --------------------------------------------------------
    # View -> Purchase
    # --------------------------------------------------------

    df["view_to_purchase_rate"] = np.where(
        df["total_views"] > 0,
        df["total_purchases"] / df["total_views"],
        0
    )

    # --------------------------------------------------------
    # Purchase rate over all events
    # --------------------------------------------------------

    df["purchase_event_rate"] = np.where(
        df["total_events"] > 0,
        df["total_purchases"] / df["total_events"],
        0
    )

    # --------------------------------------------------------
    # Cart rate over all events
    # --------------------------------------------------------

    df["cart_event_rate"] = np.where(
        df["total_events"] > 0,
        df["total_carts"] / df["total_events"],
        0
    )

    # --------------------------------------------------------
    # View rate over all events
    # --------------------------------------------------------

    df["view_event_rate"] = np.where(
        df["total_events"] > 0,
        df["total_views"] / df["total_events"],
        0
    )

    return df


# ============================================================
# 4. CREATE BRAND DATASET
# ============================================================

def create_brand_features(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE,
    max_chunks=3
):
    """
    Create the brand-level behavioral dataset.

    For your current experiment:
        max_chunks=3

    This processes approximately 15 lakh rows.
    """

    print("\n")
    print("=" * 60)
    print("PHASE 4 - BRAND FEATURE EXTRACTION")
    print("=" * 60)

    brand_chunks = []

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    for chunk_number, df in enumerate(
        read_preprocessed_chunks(
            file_path,
            chunk_size
        ),
        start=1
    ):

        print(
            f"\nAggregating brand behavior "
            f"for chunk {chunk_number}..."
        )

        brand_chunk = aggregate_brand_chunk(
            df
        )

        brand_chunks.append(
            brand_chunk
        )

        print(
            f"Brands in chunk: "
            f"{len(brand_chunk):,}"
        )

        # ----------------------------------------------------
        # Continue using only 3 chunks
        # ----------------------------------------------------

        if max_chunks is not None:

            if chunk_number >= max_chunks:
                break

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print("\nCombining brand-level results...")

    brand_df = combine_brand_chunks(
        brand_chunks
    )

    # --------------------------------------------------------
    # Behavioral rates
    # --------------------------------------------------------

    print(
        "Creating brand behavioral rates..."
    )

    brand_df = create_brand_behavior_rates(
        brand_df
    )

    # --------------------------------------------------------
    # Remove unknown brand from anomaly analysis
    # --------------------------------------------------------

    identifiable_brands = (
        brand_df[
            brand_df["brand"] != "unknown"
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    identifiable_brands.to_csv(
        BRAND_DATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PHASE 4 COMPLETED")
    print("=" * 60)

    print(
        f"\nTotal brands including unknown: "
        f"{len(brand_df):,}"
    )

    print(
        f"Identifiable brands: "
        f"{len(identifiable_brands):,}"
    )

    print(
        f"\nSaved to:"
        f"\n{BRAND_DATA_PATH}"
    )

    return identifiable_brands