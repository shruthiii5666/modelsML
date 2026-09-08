# ============================================================
# recommendation/evaluation.py
# Computes Recommendation Utility and Brand Safety Metrics
# ============================================================

import pandas as pd
import numpy as np


def evaluate_topk_session_metrics(topk_df, k=10):
    """
    Compute session-level and global average metrics for a specific Top-K recommendation dataframe:
    - Utility: HitRate@K, Precision@K, Recall@K, NDCG@K
    - Trust / Safety: Average Trust@K (ATS@K), High-Risk Exposure Rate (HRER@K), Low-Trust Exposure Rate
    """
    df = topk_df[topk_df["rank"] <= k].copy()
    grouped = df.groupby("user_session")

    hit_rates = []
    precisions = []
    recalls = []
    ndcgs = []
    avg_trusts = []
    high_risk_counts = []
    low_trust_counts = []
    total_items = 0

    for session_id, group in grouped:
        n_items = len(group)
        total_items += n_items

        # Target indicators
        targets = group["is_ground_truth_target"].values
        n_hits = targets.sum()

        # HitRate
        hit_rates.append(1.0 if n_hits > 0 else 0.0)

        # Precision@K
        precisions.append(n_hits / float(k))

        # Recall@K (assuming in-session targets)
        # Session total targets is at least max(1, n_hits)
        recalls.append(n_hits / float(max(1, n_hits)))

        # NDCG@K
        dcg = 0.0
        for idx, hit in enumerate(targets):
            if hit:
                dcg += 1.0 / np.log2(idx + 2)  # idx 0 is rank 1 -> log2(2)=1

        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, max(1, int(n_hits))))])
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # Trust Metrics
        avg_trusts.append(group["trust_score"].mean())

        # High-Risk / Low-Trust
        n_high = (group["risk_level"] == "High").sum()
        n_low_trust = (group["trust_category"] == "Low Trust").sum()
        high_risk_counts.append(n_high)
        low_trust_counts.append(n_low_trust)

    total_high_risk = sum(high_risk_counts)
    total_low_trust = sum(low_trust_counts)

    metrics = {
        f"HitRate@{k}": float(np.mean(hit_rates)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"AverageTrust@{k}": float(np.mean(avg_trusts)),
        f"HighRiskExposureRate@{k} (%)": float(total_high_risk / max(1, total_items) * 100.0),
        f"LowTrustExposureRate@{k} (%)": float(total_low_trust / max(1, total_items) * 100.0)
    }

    return metrics


def compare_all_models(model_topk_dict, k_values=[5, 10]):
    """
    Evaluate multiple model recommendation dataframes across multiple K values.
    Returns a clean comparison DataFrame.
    """
    rows = []

    for model_name, topk_df in model_topk_dict.items():
        row = {"Model": model_name}
        for k in k_values:
            m = evaluate_topk_session_metrics(topk_df, k=k)
            row.update(m)
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    return comparison_df


def analyze_rank_shifts(purchase_only_topk, trust_aware_topk, k=10):
    """
    Analyze rank position changes for individual products between Purchase-Only and Trust-Aware:
    Demonstrates how Low-Trust / High-Risk products are demoted, while High-Trust products rise.
    """
    # Merge on user_session and product_id
    p_only = purchase_only_topk[["user_session", "product_id", "brand", "risk_level", "trust_score", "rank"]].rename(
        columns={"rank": "rank_purchase_only"}
    )
    t_aware = trust_aware_topk[["user_session", "product_id", "rank"]].rename(
        columns={"rank": "rank_trust_aware"}
    )

    merged = p_only.merge(t_aware, on=["user_session", "product_id"], how="inner")

    # Shift: positive means demoted (e.g. rank 1 -> rank 8: shift = +7)
    merged["rank_shift"] = merged["rank_trust_aware"] - merged["rank_purchase_only"]

    # Calculate statistics per risk level
    shift_summary = merged.groupby("risk_level").agg(
        total_occurrences=("product_id", "count"),
        mean_rank_purchase_only=("rank_purchase_only", "mean"),
        mean_rank_trust_aware=("rank_trust_aware", "mean"),
        mean_demotion_shift=("rank_shift", "mean"),
        demoted_count=("rank_shift", lambda x: (x > 0).sum()),
        promoted_count=("rank_shift", lambda x: (x < 0).sum()),
        unchanged_count=("rank_shift", lambda x: (x == 0).sum())
    ).reset_index()

    shift_summary["demotion_rate (%)"] = (
        shift_summary["demoted_count"] / shift_summary["total_occurrences"] * 100.0
    ).round(2)

    return merged, shift_summary


def extract_demonstration_case_studies(merged_shifts_df, n_cases=5):
    """
    Extract specific, illustrative user session examples showing before-and-after rank inversions.
    """
    # Look for sessions where High-Risk brands had high initial rank in purchase-only
    high_risk_cases = merged_shifts_df[
        (merged_shifts_df["risk_level"] == "High") &
        (merged_shifts_df["rank_purchase_only"] <= 3)
    ].sort_values("rank_shift", ascending=False)

    sample_sessions = high_risk_cases["user_session"].unique()[:n_cases]
    if len(sample_sessions) == 0:
        # Fallback to medium/low trust shifts
        sample_sessions = merged_shifts_df.sort_values("rank_shift", ascending=False)["user_session"].unique()[:n_cases]

    case_studies = merged_shifts_df[merged_shifts_df["user_session"].isin(sample_sessions)].sort_values(
        ["user_session", "rank_trust_aware"]
    )
    return case_studies
