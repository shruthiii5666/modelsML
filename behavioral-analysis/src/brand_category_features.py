# ============================================================
# brand_category_features.py
# Phase 5 - Brand × Category Feature Extraction
# ============================================================

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    CHUNK_SIZE,
    BRAND_CATEGORY_DATA_PATH
)

from src.data_loader import read_preprocessed_chunks


# ============================================================
# 1. AGGREGATE BRAND × CATEGORY FOR ONE CHUNK
# ============================================================

def aggregate_brand_category_chunk(df):

    # Remove unknown brands from trust analysis
    df = df[df["brand"] != "unknown"].copy()

    # Replace missing category with unknown
    df["category_code"] = (
        df["category_code"]
        .fillna("unknown")
        .astype(str)
    )

    brand_category_df = (
        df.groupby(
            ["brand", "category_code"]
        )
        .agg(
            total_events=("event_type", "size"),

            total_views=("is_view", "sum"),

            total_carts=("is_cart", "sum"),

            total_purchases=("is_purchase", "sum"),

            unique_sessions=("user_session", "nunique"),

            unique_users=("user_id", "nunique"),

            unique_products=("product_id", "nunique"),

            average_price=("price", "mean"),

            total_revenue=("price", "sum")
        )
        .reset_index()
    )

    return brand_category_df


# ============================================================
# 2. COMBINE CHUNK RESULTS
# ============================================================

def combine_brand_category_chunks(
    brand_category_chunks
):

    if not brand_category_chunks:
        return pd.DataFrame()

    combined = pd.concat(
        brand_category_chunks,
        ignore_index=True
    )

    final_df = (
        combined
        .groupby(
            ["brand", "category_code"]
        )
        .agg(
            total_events=("total_events", "sum"),

            total_views=("total_views", "sum"),

            total_carts=("total_carts", "sum"),

            total_purchases=("total_purchases", "sum"),

            # These are summed for the prototype.
            # Exact global distinct counts will be
            # improved before final research results.
            unique_sessions=("unique_sessions", "sum"),

            unique_users=("unique_users", "sum"),

            unique_products=("unique_products", "sum"),

            average_price=("average_price", "mean"),

            total_revenue=("total_revenue", "sum")
        )
        .reset_index()
    )

    return final_df


# ============================================================
# 3. CREATE BEHAVIORAL RATES
# ============================================================

def create_behavior_rates(df):

    # --------------------------------------------------------
    # View → Cart
    # --------------------------------------------------------

    df["view_to_cart_rate"] = np.where(
        df["total_views"] > 0,
        df["total_carts"] / df["total_views"],
        0
    )

    # --------------------------------------------------------
    # Cart → Purchase
    # --------------------------------------------------------

    df["cart_to_purchase_rate"] = np.where(
        df["total_carts"] > 0,
        df["total_purchases"] / df["total_carts"],
        0
    )

    # --------------------------------------------------------
    # View → Purchase
    # --------------------------------------------------------

    df["view_to_purchase_rate"] = np.where(
        df["total_views"] > 0,
        df["total_purchases"] / df["total_views"],
        0
    )

    # --------------------------------------------------------
    # Purchase Event Rate
    # --------------------------------------------------------

    df["purchase_event_rate"] = np.where(
        df["total_events"] > 0,
        df["total_purchases"] / df["total_events"],
        0
    )

    # --------------------------------------------------------
    # Cart Event Rate
    # --------------------------------------------------------

    df["cart_event_rate"] = np.where(
        df["total_events"] > 0,
        df["total_carts"] / df["total_events"],
        0
    )

    # --------------------------------------------------------
    # View Event Rate
    # --------------------------------------------------------

    df["view_event_rate"] = np.where(
        df["total_events"] > 0,
        df["total_views"] / df["total_events"],
        0
    )

    return df


# ============================================================
# 4. MAIN PHASE 5 FUNCTION
# ============================================================

def create_brand_category_features(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE,
    max_chunks=3
):

    print("\n")
    print("=" * 60)
    print("PHASE 5 - BRAND × CATEGORY FEATURE EXTRACTION")
    print("=" * 60)

    print("\nIMPORTANT:")
    print("Only the first 3 chunks will be processed.")
    print("Chunk size:", f"{chunk_size:,}")
    print("Maximum rows:", f"{chunk_size * max_chunks:,}")

    brand_category_chunks = []

    # --------------------------------------------------------
    # Process only first 3 chunks
    # --------------------------------------------------------

    for chunk_number, df in enumerate(
        read_preprocessed_chunks(
            file_path,
            chunk_size
        ),
        start=1
    ):

        print("\n" + "-" * 60)

        print(
            f"Processing chunk {chunk_number}..."
        )

        print(
            f"Cleaned rows: {len(df):,}"
        )

        # ----------------------------------------------------
        # Aggregate brand × category
        # ----------------------------------------------------

        brand_category_chunk = (
            aggregate_brand_category_chunk(df)
        )

        brand_category_chunks.append(
            brand_category_chunk
        )

        print(
            "Brand × category combinations:",
            f"{len(brand_category_chunk):,}"
        )

        # ----------------------------------------------------
        # STOP AFTER FIRST 3 CHUNKS
        # ----------------------------------------------------

        if max_chunks is not None:

            if chunk_number >= max_chunks:

                print(
                    "\nReached maximum chunk limit."
                )

                break

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("COMBINING BRAND × CATEGORY RESULTS")
    print("=" * 60)

    final_df = combine_brand_category_chunks(
        brand_category_chunks
    )

    # --------------------------------------------------------
    # Create behavioral rates
    # --------------------------------------------------------

    print("\nCreating behavioral rates...")

    final_df = create_behavior_rates(
        final_df
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        [
            "brand",
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
    # Save
    # --------------------------------------------------------

    final_df.to_csv(
        BRAND_CATEGORY_DATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PHASE 5 COMPLETED")
    print("=" * 60)

    print(
        "\nTotal brand × category combinations:",
        f"{len(final_df):,}"
    )

    print(
        "\nSaved to:"
    )

    print(
        BRAND_CATEGORY_DATA_PATH
    )

    print("\nColumns:")

    for column in final_df.columns:
        print("  -", column)

    return final_df