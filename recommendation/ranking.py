# ============================================================
# recommendation/ranking.py
# Implements Trust-Aware Ranking and Comparative Baselines
# ============================================================

import pandas as pd
import numpy as np

from recommendation.config import (
    ALPHA,
    BETA,
    RISK_PENALTIES,
    DEFAULT_K
)


def compute_trust_aware_scores(
    candidates_df,
    alpha=ALPHA,
    beta=BETA,
    risk_penalties=RISK_PENALTIES
):
    """
    Compute proposed Trust-Aware Recommendation Score:
    Score = BaseRelevance * (1.0 + alpha * Purchase_Prob) * (Trust_Score ** beta) * Psi(Risk)
    """
    df = candidates_df.copy()

    # Intent multiplier: (1.0 + alpha * purchase_probability)
    intent_mult = 1.0 + alpha * df["purchase_probability"]

    # Trust multiplier: (trust_score ** beta)
    trust_mult = np.power(df["trust_score"], beta)

    # Discrete risk penalty multiplier Psi(b)
    penalty_mult = df["risk_level"].map(risk_penalties).fillna(0.75)

    df["trust_aware_score"] = df["base_relevance_score"] * intent_mult * trust_mult * penalty_mult
    return df


def compute_baseline_scores(candidates_df, catalog_df, alpha=ALPHA):
    """
    Compute scores for the three comparative baseline algorithms:
    1. Popularity-Only: Catalog popularity score
    2. Purchase-Only (No Trust): BaseRelevance * (1.0 + alpha * Purchase_Prob)
    3. Hard-Filter: Purchase-Only score with High-Risk brands zeroed out
    """
    df = candidates_df.copy()

    # 1. Purchase-Only Baseline (beta=0, Psi=1.0)
    intent_mult = 1.0 + alpha * df["purchase_probability"]
    df["purchase_only_score"] = df["base_relevance_score"] * intent_mult

    # 2. Hard-Filter Baseline (excludes High-Risk completely)
    df["hard_filter_score"] = df["purchase_only_score"].copy()
    high_risk_mask = (df["risk_level"] == "High")
    df.loc[high_risk_mask, "hard_filter_score"] = -1.0  # Placed below valid items

    # 3. Popularity-Only Baseline
    pop_lookup = catalog_df.set_index("product_id")["popularity_score"].to_dict()
    df["popularity_score"] = df["product_id"].map(pop_lookup).fillna(0.0)

    return df


def extract_topk_recommendations(df, score_column, model_name="trust_aware", k=DEFAULT_K):
    """
    Sort candidates by score descending per session and select Top-K.
    """
    # Sort by session and score descending
    sorted_df = df.sort_values(
        ["user_session", score_column, "base_relevance_score"],
        ascending=[True, False, False]
    ).copy()

    # Assign rank 1 to K
    sorted_df["rank"] = sorted_df.groupby("user_session").cumcount() + 1
    topk = sorted_df[sorted_df["rank"] <= k].copy()
    topk["model_name"] = model_name

    return topk
