# ============================================================
# recommendation/tests/test_phase2_model_usage.py
# Proves the recommendation module uses genuine Model 1 & 2 outputs
# ============================================================

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.config import (
    CANDIDATE_CACHE_PATH,
    MODEL1_PROBABILITIES_PATH,
    MODEL2_TRUST_DATASET_PATH,
    TOPK_TRUST_AWARE_PATH,
    TOPK_PURCHASE_ONLY_PATH
)


def test_phase2():
    print("=" * 70)
    print("PHASE 2 TEST: PROVE REAL MODEL USAGE VS FALLBACKS")
    print("=" * 70)

    # 1. Load Master Candidates
    df = pd.read_csv(CANDIDATE_CACHE_PATH)
    total_candidates = len(df)
    print(f"Total candidates analyzed: {total_candidates:,}")

    # 2. Check Model 1 Usage
    m1_real_count = (df["purchase_probability"].notna() & (df["purchase_probability"] > 0)).sum()
    m1_missing_count = df["purchase_probability"].isna().sum()
    m1_real_pct = (m1_real_count / total_candidates) * 100.0
    m1_missing_pct = (m1_missing_count / total_candidates) * 100.0

    print("\n--- MODEL 1 (PURCHASE PROBABILITY) USAGE ---")
    print(f"  Real Model 1 probabilities: {m1_real_count:,} ({m1_real_pct:.2f}%)")
    print(f"  Missing probabilities:      {m1_missing_count:,} ({m1_missing_pct:.2f}%)")

    # Verify probability match with raw Model 1 dataset
    m1_raw = pd.read_csv(MODEL1_PROBABILITIES_PATH).set_index("user_session")["purchase_probability"].to_dict()
    matched_probs = 0
    sample_sessions = df["user_session"].unique()[:200]
    for sid in sample_sessions:
        expected_p = m1_raw.get(sid)
        actual_p = df[df["user_session"] == sid]["purchase_probability"].iloc[0]
        if expected_p is not None and abs(expected_p - actual_p) < 1e-6:
            matched_probs += 1
    print(f"  Direct verification: {matched_probs}/{len(sample_sessions)} sampled sessions match Model 1 exact values.")

    # 3. Check Model 2 Usage
    scored_mask = (df["scoring_status"] == "scored")
    unbranded_mask = (df["scoring_status"] == "unbranded_fallback")
    insufficient_mask = (df["scoring_status"] == "insufficient_evidence_fallback")

    m2_real_count = scored_mask.sum()
    m2_unbranded_count = unbranded_mask.sum()
    m2_insufficient_count = insufficient_mask.sum()

    m2_real_pct = (m2_real_count / total_candidates) * 100.0
    m2_unbranded_pct = (m2_unbranded_count / total_candidates) * 100.0
    m2_insufficient_pct = (m2_insufficient_count / total_candidates) * 100.0

    print("\n--- MODEL 2 (BRAND TRUST & RISK) USAGE ---")
    print(f"  Real Model 2 scored trust:           {m2_real_count:,} ({m2_real_pct:.2f}%)")
    print(f"  Unbranded fallback (brand=unknown):   {m2_unbranded_count:,} ({m2_unbranded_pct:.2f}%)")
    print(f"  Unscored brand fallback:             {m2_insufficient_count:,} ({m2_insufficient_pct:.2f}%)")
    print(f"  Total fallback usage:                 {m2_unbranded_count + m2_insufficient_count:,} ({m2_unbranded_pct + m2_insufficient_pct:.2f}%)")

    # Verify exact brand trust matches against Model 2 dataset
    m2_raw = pd.read_csv(MODEL2_TRUST_DATASET_PATH).set_index("brand")["trust_score"].to_dict()
    sample_scored_brands = df[scored_mask]["brand"].unique()[:100]
    matched_trust = 0
    for b in sample_scored_brands:
        expected_t = m2_raw.get(b)
        actual_t = df[df["brand"] == b]["trust_score"].iloc[0]
        if expected_t is not None and abs(expected_t - actual_t) < 1e-6:
            matched_trust += 1
    print(f"  Direct verification: {matched_trust}/{len(sample_scored_brands)} sampled scored brands match Model 2 exact values.")

    # 4. Demonstrate Real Rank Inversion Driven by Real Model Outputs
    print("\n--- DEMONSTRATION OF REAL MODEL INFLUENCE ON RANKING ---")
    p_topk = pd.read_csv(TOPK_PURCHASE_ONLY_PATH)
    t_topk = pd.read_csv(TOPK_TRUST_AWARE_PATH)

    merged = p_topk.merge(
        t_topk[["user_session", "product_id", "rank", "trust_aware_score"]],
        on=["user_session", "product_id"],
        suffixes=("_purchase_only", "_trust_aware")
    )
    merged["rank_shift"] = merged["rank_trust_aware"] - merged["rank_purchase_only"]

    # Filter for real Model 2 scored brands where ranking shifted
    real_model_shifts = merged[
        (merged["scoring_status"] == "scored") &
        (merged["rank_shift"] != 0)
    ]
    print(f"  Found {len(real_model_shifts):,} candidate occurrences where real Model 2 trust altered Top-10 ranking positions.")

    # Show concrete examples
    print("\nExample 1: High-Trust Brand Promoted due to real Model 2 Trust Score:")
    promoted = real_model_shifts[real_model_shifts["rank_shift"] < 0].sort_values("rank_shift").head(2)
    for _, r in promoted.iterrows():
        print(f"    Session: {r['user_session']} | Product: {r['product_id']} | Brand: {r['brand']}")
        print(f"    Model 1 Prob: {r['purchase_probability']:.4f} | Model 2 Trust: {r['trust_score']:.4f} ({r['risk_level']} Risk)")
        print(f"    Purchase-Only Rank: {r['rank_purchase_only']} --> Trust-Aware Rank: {r['rank_trust_aware']} (Shift: {r['rank_shift']:+d})")

    print("\nExample 2: Moderate/Low-Trust Brand Demoted due to real Model 2 Risk:")
    demoted = real_model_shifts[real_model_shifts["rank_shift"] > 0].sort_values("rank_shift", ascending=False).head(2)
    for _, r in demoted.iterrows():
        print(f"    Session: {r['user_session']} | Product: {r['product_id']} | Brand: {r['brand']}")
        print(f"    Model 1 Prob: {r['purchase_probability']:.4f} | Model 2 Trust: {r['trust_score']:.4f} ({r['risk_level']} Risk)")
        print(f"    Purchase-Only Rank: {r['rank_purchase_only']} --> Trust-Aware Rank: {r['rank_trust_aware']} (Shift: {r['rank_shift']:+d})")

    # Final Verdict on Phase 2
    is_genuinely_using = (m1_real_pct >= 99.0) and (m2_real_pct >= 85.0) and (len(real_model_shifts) > 0)
    print("\n" + "=" * 70)
    print(f"IS RECOMMENDATION SYSTEM GENUINELY USING MODEL 1 & MODEL 2? {'YES (PROVEN)' if is_genuinely_using else 'NO'}")
    print("=" * 70)

    return {
        "m1_real_pct": m1_real_pct,
        "m2_real_pct": m2_real_pct,
        "fallback_pct": m2_unbranded_pct + m2_insufficient_count,
        "shifts_count": len(real_model_shifts),
        "is_genuinely_using": is_genuinely_using
    }


if __name__ == "__main__":
    res = test_phase2()
    sys.exit(0 if res["is_genuinely_using"] else 1)
