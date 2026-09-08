# ============================================================
# recommendation/run_recommendation.py
# End-to-End Orchestrator for Trust-Aware Recommendation Module
# ============================================================

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from recommendation.config import (
    RAW_CLICKSTREAM_PATH,
    PRODUCT_CATALOG_PATH,
    CANDIDATE_CACHE_PATH,
    TOPK_TRUST_AWARE_PATH,
    TOPK_PURCHASE_ONLY_PATH,
    MODEL_COMPARISON_PATH,
    SENSITIVITY_ANALYSIS_PATH,
    BEFORE_AFTER_CASES_PATH,
    PLOT_TRADEOFF_PATH,
    PLOT_HIGH_RISK_PATH,
    PLOT_RANK_DISPLACEMENT_PATH,
    DEFAULT_K,
    EVALUATION_K_VALUES,
    ALPHA,
    BETA,
    BETA_SWEEP
)
from recommendation.candidate_generator import (
    build_or_load_catalog,
    extract_session_interactions,
    generate_candidate_pool
)
from recommendation.trust_integrator import (
    load_purchase_intent_data,
    load_brand_trust_data,
    integrate_candidate_pool
)
from recommendation.ranking import (
    compute_trust_aware_scores,
    compute_baseline_scores,
    extract_topk_recommendations
)
from recommendation.evaluation import (
    compare_all_models,
    analyze_rank_shifts,
    extract_demonstration_case_studies
)


def generate_visualizations(comparison_df, sensitivity_df, shifts_df):
    """
    Generate clean, 300-DPI visual plots for the FYP dissertation and viva presentation.
    """
    print("\nGenerating publication-quality visualization plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # ------------------------------------------------------------
    # Plot 1: Trust vs Utility Trade-off across Beta
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    color_trust = "#1a73e8"
    color_ndcg = "#d93025"

    ax1.set_xlabel(r"Trust Sensitivity Parameter ($\beta$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Trust@10", color=color_trust, fontsize=12, fontweight="bold")
    line1 = ax1.plot(
        sensitivity_df["Beta"],
        sensitivity_df["AverageTrust@10"],
        color=color_trust,
        marker="o",
        linewidth=2.5,
        label="Average Trust@10"
    )
    ax1.tick_params(axis="y", labelcolor=color_trust)
    ax1.set_ylim(0.70, 1.00)

    ax2 = ax1.twinx()
    ax2.set_ylabel("NDCG@10 (Recommendation Utility)", color=color_ndcg, fontsize=12, fontweight="bold")
    line2 = ax2.plot(
        sensitivity_df["Beta"],
        sensitivity_df["NDCG@10"],
        color=color_ndcg,
        marker="s",
        linestyle="--",
        linewidth=2.5,
        label="NDCG@10"
    )
    ax2.tick_params(axis="y", labelcolor=color_ndcg)

    plt.title("Trust-Utility Trade-off across Trust Sensitivity Parameter (Beta)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(PLOT_TRADEOFF_PATH, dpi=300)
    plt.close()
    print(f"  Saved: {PLOT_TRADEOFF_PATH}")

    # ------------------------------------------------------------
    # Plot 2: High-Risk Exposure Rate Comparison
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    models = comparison_df["Model"].tolist()
    rates_k10 = comparison_df["HighRiskExposureRate@10 (%)"].tolist()
    rates_k5 = comparison_df["HighRiskExposureRate@5 (%)"].tolist()

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width / 2, rates_k10, width, label="Top-10 Exposure", color="#e53935", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width / 2, rates_k5, width, label="Top-5 Exposure", color="#fb8c00", edgecolor="black", alpha=0.85)

    ax.set_ylabel("High-Risk Exposure Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Exposure to High-Risk Brands across Recommendation Models", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.legend(frameon=True)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    plt.savefig(PLOT_HIGH_RISK_PATH, dpi=300)
    plt.close()
    print(f"  Saved: {PLOT_HIGH_RISK_PATH}")

    # ------------------------------------------------------------
    # Plot 3: Rank Displacement Distribution by Risk Level
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    risk_groups = shifts_df.groupby("risk_level")["rank_shift"].mean()
    colors = ["#43a047" if r == "Low" else "#fb8c00" if r == "Medium" else "#e53935" for r in risk_groups.index]

    bars = ax.bar(risk_groups.index, risk_groups.values, color=colors, edgecolor="black", width=0.5, alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel("Average Rank Demotion Shift (+ positions)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Brand Risk Level", fontsize=12, fontweight="bold")
    ax.set_title("Average Rank Shift from Purchase-Only to Trust-Aware Recommendation", fontsize=13, fontweight="bold", pad=12)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:+.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4 if h >= 0 else -14), textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    fig.tight_layout()
    plt.savefig(PLOT_RANK_DISPLACEMENT_PATH, dpi=300)
    plt.close()
    print(f"  Saved: {PLOT_RANK_DISPLACEMENT_PATH}")


def run_recommendation_pipeline(max_test_sessions=1000):
    """
    Execute the entire recommendation, ranking, baseline comparison, and evaluation pipeline.
    """
    print("\n" + "=" * 75)
    print("TRUST-AWARE RECOMMENDATION MODULE PIPELINE")
    print("=" * 75)

    # 1. Build / Load Product Catalog
    print("\n[STEP 1/7] Initializing Product Catalog...")
    catalog_df = build_or_load_catalog(raw_path=RAW_CLICKSTREAM_PATH)

    # 2. Load Model 1 and Model 2 Outputs
    print("\n[STEP 2/7] Loading Completed Model Outputs...")
    intent_df = load_purchase_intent_data()
    trust_df = load_brand_trust_data()

    # 3. Candidate Generation
    print("\n[STEP 3/7] Generating Recommendation Candidates...")
    interactions_df = extract_session_interactions(raw_path=RAW_CLICKSTREAM_PATH)
    test_sessions = intent_df["user_session"].unique()[:max_test_sessions]

    raw_candidates_df = generate_candidate_pool(
        target_sessions=test_sessions,
        catalog_df=catalog_df,
        interactions_df=interactions_df
    )

    # 4. Integrate Candidates with ML Model Outputs & Fallbacks
    print("\n[STEP 4/7] Integrating Model Intent + Brand Trust Scores...")
    integrated_df = integrate_candidate_pool(
        candidates_df=raw_candidates_df,
        intent_df=intent_df,
        trust_df=trust_df
    )
    integrated_df.to_csv(CANDIDATE_CACHE_PATH, index=False)
    print(f"Saved master candidates dataset to: {CANDIDATE_CACHE_PATH}")

    # 5. Compute Scores for Proposed Model and Baselines
    print("\n[STEP 5/7] Ranking Candidates across Models...")
    # Proposed Trust-Aware Model
    scored_df = compute_trust_aware_scores(integrated_df, alpha=ALPHA, beta=BETA)

    # Baselines
    scored_df = compute_baseline_scores(scored_df, catalog_df=catalog_df, alpha=ALPHA)

    # Extract Top-K recommendations
    topk_trust_aware = extract_topk_recommendations(
        scored_df, score_column="trust_aware_score", model_name="Trust-Aware (Proposed)", k=DEFAULT_K
    )
    topk_purchase_only = extract_topk_recommendations(
        scored_df, score_column="purchase_only_score", model_name="Purchase-Only Baseline", k=DEFAULT_K
    )
    topk_hard_filter = extract_topk_recommendations(
        scored_df, score_column="hard_filter_score", model_name="Hard-Filter Safety Baseline", k=DEFAULT_K
    )
    topk_popularity = extract_topk_recommendations(
        scored_df, score_column="popularity_score", model_name="Popularity-Only Baseline", k=DEFAULT_K
    )

    topk_trust_aware.to_csv(TOPK_TRUST_AWARE_PATH, index=False)
    topk_purchase_only.to_csv(TOPK_PURCHASE_ONLY_PATH, index=False)
    print(f"Saved Trust-Aware Top-K recommendations to: {TOPK_TRUST_AWARE_PATH}")
    print(f"Saved Purchase-Only Top-K recommendations to: {TOPK_PURCHASE_ONLY_PATH}")

    # 6. Comparative Evaluation across All Models
    print("\n[STEP 6/7] Evaluating Recommendation Utility & Trust/Safety...")
    model_dict = {
        "Popularity Baseline": topk_popularity,
        "Purchase-Only Baseline": topk_purchase_only,
        "Hard-Filter Safety Baseline": topk_hard_filter,
        "Trust-Aware (Proposed)": topk_trust_aware
    }
    comparison_df = compare_all_models(model_dict, k_values=EVALUATION_K_VALUES)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"Saved model comparison table to: {MODEL_COMPARISON_PATH}")

    print("\n" + "=" * 75)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 75)
    print(comparison_df.to_string(index=False))

    # Sensitivity Analysis across Beta Sweep
    print("\nRunning Sensitivity Analysis across Beta sweep...")
    sensitivity_rows = []
    for b in BETA_SWEEP:
        b_df = compute_trust_aware_scores(integrated_df, alpha=ALPHA, beta=b)
        b_topk = extract_topk_recommendations(b_df, score_column="trust_aware_score", k=DEFAULT_K)
        from recommendation.evaluation import evaluate_topk_session_metrics
        m = evaluate_topk_session_metrics(b_topk, k=DEFAULT_K)
        m["Beta"] = b
        sensitivity_rows.append(m)

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    # Order columns with Beta first
    cols = ["Beta"] + [c for c in sensitivity_df.columns if c != "Beta"]
    sensitivity_df = sensitivity_df[cols]
    sensitivity_df.to_csv(SENSITIVITY_ANALYSIS_PATH, index=False)
    print(f"Saved sensitivity analysis table to: {SENSITIVITY_ANALYSIS_PATH}")
    print(sensitivity_df.to_string(index=False))

    # Rank Shifts & Before vs After Demonstration Cases
    print("\n[STEP 7/7] Generating Before vs After Case Studies & Visualizations...")
    merged_shifts, shift_summary = analyze_rank_shifts(topk_purchase_only, topk_trust_aware, k=DEFAULT_K)
    print("\nRank Demotion Summary by Risk Level:")
    print(shift_summary.to_string(index=False))

    cases_df = extract_demonstration_case_studies(merged_shifts, n_cases=5)
    cases_df.to_csv(BEFORE_AFTER_CASES_PATH, index=False)
    print(f"Saved before-after case studies to: {BEFORE_AFTER_CASES_PATH}")

    # Generate visual plots
    generate_visualizations(comparison_df, sensitivity_df, merged_shifts)

    print("\n" + "=" * 75)
    print("RECOMMENDATION MODULE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 75)
    return comparison_df, sensitivity_df, cases_df


if __name__ == "__main__":
    run_recommendation_pipeline()
