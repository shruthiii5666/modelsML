# ============================================================
# recommendation/candidate_generator.py
# Generates in-session and category-affinity product candidates
# ============================================================

import os
import pandas as pd
import numpy as np
from pathlib import Path

from recommendation.config import (
    RAW_CLICKSTREAM_PATH,
    PRODUCT_CATALOG_PATH,
    IN_SESSION_CART_RELEVANCE,
    IN_SESSION_VIEW_RELEVANCE,
    CATEGORY_AFFINITY_MIN_RELEVANCE,
    CATEGORY_AFFINITY_MAX_RELEVANCE,
    MAX_CATEGORY_CANDIDATES_PER_SESSION,
    SAMPLE_CATALOG_ROWS
)


def build_or_load_catalog(
    raw_path=RAW_CLICKSTREAM_PATH,
    catalog_path=PRODUCT_CATALOG_PATH,
    max_rows=SAMPLE_CATALOG_ROWS,
    force_rebuild=False
):
    """
    Extract or load a lightweight product catalog:
    (product_id, category_code, brand, price, popularity_score)
    """
    if not force_rebuild and os.path.exists(catalog_path):
        print(f"Loading cached product catalog from: {catalog_path}")
        return pd.read_csv(catalog_path)

    print(f"Building product catalog from clickstream ({max_rows:,} rows)...")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Clickstream file not found at: {raw_path}")

    df = pd.read_csv(
        raw_path,
        nrows=max_rows,
        usecols=["event_type", "product_id", "category_code", "brand", "price"]
    )

    # Clean brand & category
    df["brand"] = df["brand"].fillna("unknown").astype(str).str.strip().str.lower()
    df.loc[df["brand"] == "", "brand"] = "unknown"

    df["category_code"] = df["category_code"].fillna("unknown").astype(str).str.strip()
    df.loc[df["category_code"] == "", "category_code"] = "unknown"

    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    # Deduplicate canonical metadata per product_id (take mode / median)
    meta = (
        df.groupby("product_id")
        .agg(
            category_code=("category_code", lambda x: x.mode()[0] if not x.empty else "unknown"),
            brand=("brand", lambda x: x.mode()[0] if not x.empty else "unknown"),
            price=("price", "median"),
            total_events=("event_type", "size"),
            views=("event_type", lambda x: (x == "view").sum()),
            carts=("event_type", lambda x: (x == "cart").sum()),
            purchases=("event_type", lambda x: (x == "purchase").sum())
        )
        .reset_index()
    )

    # Calculate global normalized popularity score [0.0, 1.0]
    raw_pop = meta["views"] + 3.0 * meta["carts"] + 5.0 * meta["purchases"]
    meta["popularity_score"] = (raw_pop - raw_pop.min()) / (raw_pop.max() - raw_pop.min() + 1e-9)

    meta.to_csv(catalog_path, index=False)
    print(f"Saved product catalog ({len(meta):,} items) to: {catalog_path}")
    return meta


def extract_session_interactions(raw_path=RAW_CLICKSTREAM_PATH, max_rows=SAMPLE_CATALOG_ROWS):
    """
    Extract session event logs to construct in-session candidate pools and ground-truth targets.
    """
    print(f"Extracting session interactions from {raw_path}...")
    df = pd.read_csv(
        raw_path,
        nrows=max_rows,
        usecols=["event_time", "event_type", "product_id", "category_code", "brand", "price", "user_session"]
    )

    df = df.dropna(subset=["user_session"])
    df["brand"] = df["brand"].fillna("unknown").astype(str).str.strip().str.lower()
    df["category_code"] = df["category_code"].fillna("unknown").astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    return df


def generate_candidate_pool(
    target_sessions,
    catalog_df,
    interactions_df,
    max_category_candidates=MAX_CATEGORY_CANDIDATES_PER_SESSION
):
    """
    For every session in target_sessions, generate:
    1. In-session candidates (viewed / carted products)
    2. Category-affinity candidates (top items in browsed categories)
    Also identifies ground-truth target items (carted or purchased in the session).
    """
    print(f"Generating candidate pools for {len(target_sessions):,} sessions...")

    # Filter interactions to target sessions
    session_events = interactions_df[interactions_df["user_session"].isin(target_sessions)].copy()

    # Pre-index catalog by category_code for ultra-fast category candidate lookup
    category_top_items = {}
    for cat, group in catalog_df.groupby("category_code"):
        top_items = group.sort_values("popularity_score", ascending=False).head(max_category_candidates)
        category_top_items[cat] = top_items

    catalog_lookup = catalog_df.set_index("product_id").to_dict(orient="index")

    candidate_records = []

    # Group interactions by user_session
    grouped = session_events.groupby("user_session")

    for session_id, events in grouped:
        in_session_products = set()
        categories_seen = set()

        # Identify ground-truth conversion targets (cart or purchase in session)
        cart_or_purchase = set(events[events["event_type"].isin(["cart", "purchase"])]["product_id"])
        # If no cart/purchase, the last viewed product is the positive target
        if not cart_or_purchase and not events.empty:
            cart_or_purchase = {events.iloc[-1]["product_id"]}

        # 1. In-Session Candidates
        for _, row in events.iterrows():
            pid = row["product_id"]
            if pid in in_session_products:
                continue
            in_session_products.add(pid)

            cat = row["category_code"]
            if cat != "unknown":
                categories_seen.add(cat)

            # Assign in-session base relevance
            is_cart = (row["event_type"] == "cart")
            rel_score = IN_SESSION_CART_RELEVANCE if is_cart else IN_SESSION_VIEW_RELEVANCE

            meta = catalog_lookup.get(pid, {})
            brand_val = row["brand"] if row["brand"] != "unknown" else meta.get("brand", "unknown")
            price_val = row["price"] if row["price"] > 0 else meta.get("price", 0.0)

            candidate_records.append({
                "user_session": session_id,
                "product_id": pid,
                "category_code": cat,
                "brand": brand_val,
                "price": price_val,
                "candidate_source": "in_session",
                "base_relevance_score": rel_score,
                "is_ground_truth_target": 1 if pid in cart_or_purchase else 0
            })

        # 2. Category-Affinity Candidates (expansion to ensure at least K candidates)
        if not categories_seen:
            # Fallback to general popular categories if only unknown category was seen
            categories_seen = {"electronics.smartphone", "appliances.environment.water_heater"}

        added_cat_count = 0
        for cat in categories_seen:
            top_cat_items = category_top_items.get(cat)
            if top_cat_items is None:
                continue

            for _, cat_item in top_cat_items.iterrows():
                cid = int(cat_item["product_id"])
                if cid in in_session_products:
                    continue
                in_session_products.add(cid)

                # Scale affinity relevance based on popularity
                pop = cat_item["popularity_score"]
                aff_score = CATEGORY_AFFINITY_MIN_RELEVANCE + pop * (CATEGORY_AFFINITY_MAX_RELEVANCE - CATEGORY_AFFINITY_MIN_RELEVANCE)

                candidate_records.append({
                    "user_session": session_id,
                    "product_id": cid,
                    "category_code": cat_item["category_code"],
                    "brand": cat_item["brand"],
                    "price": cat_item["price"],
                    "candidate_source": "category_affinity",
                    "base_relevance_score": aff_score,
                    "is_ground_truth_target": 1 if cid in cart_or_purchase else 0
                })
                added_cat_count += 1
                if added_cat_count >= max_category_candidates:
                    break

    candidates_df = pd.DataFrame(candidate_records)
    print(f"Generated {len(candidates_df):,} total candidates across {len(grouped):,} sessions.")
    print(f"Average candidates per session: {len(candidates_df) / max(1, len(grouped)):.1f}")
    return candidates_df
