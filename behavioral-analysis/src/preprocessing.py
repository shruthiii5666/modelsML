# ============================================================
# preprocessing.py
# Phase 2 - Data Cleaning and Event Preparation
# ============================================================

import pandas as pd
import numpy as np

from src.config import VALID_EVENT_TYPES


# ============================================================
# 1. CLEAN A SINGLE CHUNK
# ============================================================

def preprocess_chunk(df):
    """
    Clean and prepare one chunk of the large clickstream
    dataset.

    The complete CSV is NOT loaded into memory.
    This function processes one chunk at a time.
    """

    # --------------------------------------------------------
    # Make a copy
    # --------------------------------------------------------

    df = df.copy()

    # --------------------------------------------------------
    # 1. Remove exact duplicate rows
    # --------------------------------------------------------

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # 2. Convert event_time to datetime
    # --------------------------------------------------------

    df["event_time"] = pd.to_datetime(
        df["event_time"],
        errors="coerce"
    )

    # Remove records with invalid timestamps
    df = df.dropna(
        subset=["event_time"]
    )

    # --------------------------------------------------------
    # 3. Clean event_type
    # --------------------------------------------------------

    df["event_type"] = (
        df["event_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep only valid interaction types
    df = df[
        df["event_type"].isin(
            VALID_EVENT_TYPES
        )
    ]

    # --------------------------------------------------------
    # 4. Handle missing user_session
    # --------------------------------------------------------

    # Sessions are essential for behavioral analysis.
    # Therefore records without a session ID cannot be used
    # reliably for session-level analysis.

    df = df.dropna(
        subset=["user_session"]
    )

    # Remove empty session strings
    df = df[
        df["user_session"]
        .astype(str)
        .str.strip()
        .ne("")
    ]

    # --------------------------------------------------------
    # 5. Clean brand
    # --------------------------------------------------------

    df["brand"] = (
        df["brand"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Empty brand values
    df.loc[
        df["brand"] == "",
        "brand"
    ] = "unknown"

    # --------------------------------------------------------
    # 6. Clean category_code
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
    # 7. Convert price
    # --------------------------------------------------------

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    # Prices <= 0 are considered invalid
    df.loc[
        df["price"] <= 0,
        "price"
    ] = np.nan

    # --------------------------------------------------------
    # 8. Create event indicators
    # --------------------------------------------------------

    df["is_view"] = (
        df["event_type"] == "view"
    ).astype("int8")

    df["is_cart"] = (
        df["event_type"] == "cart"
    ).astype("int8")

    df["is_purchase"] = (
        df["event_type"] == "purchase"
    ).astype("int8")

    # --------------------------------------------------------
    # 9. Sort within the chunk
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "user_session",
            "event_time"
        ]
    )

    # --------------------------------------------------------
    # 10. Reset index
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    return df