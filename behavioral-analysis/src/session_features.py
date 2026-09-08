# ============================================================
# session_features.py
# Phase 3 - Session-Level Behavioral Feature Extraction
# ============================================================

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    CHUNK_SIZE,
    SESSION_DATA_PATH,
    SESSION_BRAND_DATA_PATH
)

from src.data_loader import read_preprocessed_chunks


# ============================================================
# 1. AGGREGATE ONE CHUNK AT SESSION LEVEL
# ============================================================

def aggregate_session_chunk(df):
    """
    Aggregate one preprocessed chunk at session level.

    Each row in the output represents one user session
    observed within this chunk.
    """

    session_features = (
        df.groupby("user_session")
        .agg(
            total_events=("event_type", "size"),

            views=("is_view", "sum"),

            carts=("is_cart", "sum"),

            purchases=("is_purchase", "sum"),

            unique_products=("product_id", "nunique"),

            unique_brands=("brand", "nunique"),

            unique_categories=("category_code", "nunique"),

            first_event_time=("event_time", "min"),

            last_event_time=("event_time", "max"),

            unique_users=("user_id", "nunique")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Session duration
    # --------------------------------------------------------

    session_features["session_duration_seconds"] = (
        session_features["last_event_time"]
        - session_features["first_event_time"]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Purchase indicator
    # --------------------------------------------------------

    session_features["has_purchase"] = (
        session_features["purchases"] > 0
    ).astype("int8")

    # --------------------------------------------------------
    # Cart indicator
    # --------------------------------------------------------

    session_features["has_cart"] = (
        session_features["carts"] > 0
    ).astype("int8")

    # --------------------------------------------------------
    # View-only session
    # --------------------------------------------------------

    session_features["view_only"] = (
        (session_features["views"] > 0)
        &
        (session_features["carts"] == 0)
        &
        (session_features["purchases"] == 0)
    ).astype("int8")

    return session_features


# ============================================================
# 2. AGGREGATE SESSION-BRAND BEHAVIOR
# ============================================================

def aggregate_session_brand_chunk(df):
    """
    Aggregate behavioral activity by:

        user_session + brand

    This is important because a single session can contain
    multiple brands.
    """

    session_brand = (
        df.groupby(
            [
                "user_session",
                "brand"
            ]
        )
        .agg(
            brand_events=("event_type", "size"),

            brand_views=("is_view", "sum"),

            brand_carts=("is_cart", "sum"),

            brand_purchases=("is_purchase", "sum"),

            brand_products=("product_id", "nunique"),

            brand_categories=("category_code", "nunique"),

            first_brand_event=("event_time", "min"),

            last_brand_event=("event_time", "max")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Brand-level session duration
    # --------------------------------------------------------

    session_brand["brand_session_duration_seconds"] = (
        session_brand["last_brand_event"]
        - session_brand["first_brand_event"]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Purchase indicator
    # --------------------------------------------------------

    session_brand["brand_has_purchase"] = (
        session_brand["brand_purchases"] > 0
    ).astype("int8")

    return session_brand


# ============================================================
# 3. COMBINE SESSION CHUNK RESULTS
# ============================================================

def combine_session_features(session_chunks):
    """
    Combine partial session aggregations.

    A session may occur across more than one chunk, so
    numerical counts are summed.
    """

    if not session_chunks:
        return pd.DataFrame()

    combined = pd.concat(
        session_chunks,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Re-aggregate sessions that appeared in multiple chunks
    # --------------------------------------------------------

    combined = (
        combined.groupby("user_session")
        .agg(
            total_events=("total_events", "sum"),

            views=("views", "sum"),

            carts=("carts", "sum"),

            purchases=("purchases", "sum"),

            unique_products=("unique_products", "sum"),

            unique_brands=("unique_brands", "sum"),

            unique_categories=("unique_categories", "sum"),

            first_event_time=("first_event_time", "min"),

            last_event_time=("last_event_time", "max"),

            unique_users=("unique_users", "sum")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Recalculate duration after combining chunks
    # --------------------------------------------------------

    combined["session_duration_seconds"] = (
        combined["last_event_time"]
        - combined["first_event_time"]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Recalculate behavioral indicators
    # --------------------------------------------------------

    combined["has_purchase"] = (
        combined["purchases"] > 0
    ).astype("int8")

    combined["has_cart"] = (
        combined["carts"] > 0
    ).astype("int8")

    combined["view_only"] = (
        (combined["views"] > 0)
        &
        (combined["carts"] == 0)
        &
        (combined["purchases"] == 0)
    ).astype("int8")

    return combined


# ============================================================
# 4. COMBINE SESSION-BRAND CHUNKS
# ============================================================

def combine_session_brand_features(session_brand_chunks):
    """
    Combine partial session-brand aggregations.
    """

    if not session_brand_chunks:
        return pd.DataFrame()

    combined = pd.concat(
        session_brand_chunks,
        ignore_index=True
    )

    combined = (
        combined.groupby(
            [
                "user_session",
                "brand"
            ]
        )
        .agg(
            brand_events=("brand_events", "sum"),

            brand_views=("brand_views", "sum"),

            brand_carts=("brand_carts", "sum"),

            brand_purchases=("brand_purchases", "sum"),

            brand_products=("brand_products", "sum"),

            brand_categories=("brand_categories", "sum"),

            first_brand_event=("first_brand_event", "min"),

            last_brand_event=("last_brand_event", "max")
        )
        .reset_index()
    )

    combined["brand_session_duration_seconds"] = (
        combined["last_brand_event"]
        - combined["first_brand_event"]
    ).dt.total_seconds()

    combined["brand_has_purchase"] = (
        combined["brand_purchases"] > 0
    ).astype("int8")

    return combined


# ============================================================
# 5. CREATE SESSION DATASET
# ============================================================

def create_session_features(
    file_path=RAW_DATA_PATH,
    chunk_size=CHUNK_SIZE,
    max_chunks=None
):
    """
    Process the dataset chunk-by-chunk and create:

        session_features.csv
        session_brand_features.csv

    max_chunks:
        Used for testing.

        max_chunks=3
        -> process only first 3 chunks

        max_chunks=None
        -> process complete dataset
    """

    session_chunks = []

    session_brand_chunks = []

    print("\n")
    print("=" * 60)
    print("PHASE 3 - SESSION FEATURE EXTRACTION")
    print("=" * 60)

    for chunk_number, df in enumerate(
        read_preprocessed_chunks(
            file_path,
            chunk_size
        ),
        start=1
    ):

        print(
            f"\nCreating session features "
            f"for chunk {chunk_number}..."
        )

        # ----------------------------------------------------
        # Session-level aggregation
        # ----------------------------------------------------

        session_chunk = aggregate_session_chunk(
            df
        )

        session_chunks.append(
            session_chunk
        )

        # ----------------------------------------------------
        # Session-brand aggregation
        # ----------------------------------------------------

        session_brand_chunk = (
            aggregate_session_brand_chunk(
                df
            )
        )

        session_brand_chunks.append(
            session_brand_chunk
        )

        print(
            f"Sessions in chunk: "
            f"{len(session_chunk):,}"
        )

        print(
            f"Session-brand pairs in chunk: "
            f"{len(session_brand_chunk):,}"
        )

        # ----------------------------------------------------
        # Testing limit
        # ----------------------------------------------------

        if (
            max_chunks is not None
            and chunk_number >= max_chunks
        ):
            break

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    print("\nCombining session results...")

    final_sessions = combine_session_features(
        session_chunks
    )

    final_session_brand = (
        combine_session_brand_features(
            session_brand_chunks
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    final_sessions.to_csv(
        SESSION_DATA_PATH,
        index=False
    )

    final_session_brand.to_csv(
        SESSION_BRAND_DATA_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("SESSION FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        f"\nTotal sessions: "
        f"{len(final_sessions):,}"
    )

    print(
        f"Total session-brand pairs: "
        f"{len(final_session_brand):,}"
    )

    print(
        f"\nSession dataset saved to:"
        f"\n{SESSION_DATA_PATH}"
    )

    print(
        f"\nSession-brand dataset saved to:"
        f"\n{SESSION_BRAND_DATA_PATH}"
    )

    return (
        final_sessions,
        final_session_brand
    )