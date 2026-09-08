# ============================================================
# recommendation/tests/test_phase1_model_artifacts.py
# Rigorous validation of existing Model 1 & Model 2 artifacts
# ============================================================

import os
import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_phase1():
    print("=" * 70)
    print("PHASE 1 TEST: VERIFY EXISTING MODEL ARTIFACTS AND OUTPUTS")
    print("=" * 70)
    results = {"passed": 0, "failed": 0, "details": []}

    def assert_check(name, condition, msg=""):
        if condition:
            print(f"  [PASS] {name}")
            results["passed"] += 1
            results["details"].append({"name": name, "status": "PASS", "message": msg})
        else:
            print(f"  [FAIL] {name}: {msg}")
            results["failed"] += 1
            results["details"].append({"name": name, "status": "FAIL", "message": msg})

    # 1. Model 1 Checkpoints Exist & Loadable
    m1_models = [
        "user-intent/models/purchase_lgbm.pkl",
        "user-intent/models/purchase_xgb.pkl",
        "user-intent/models/purchase_rf.pkl",
        "user-intent/models/purchase_stacking_meta.pkl"
    ]
    for mpath in m1_models:
        full_path = PROJECT_ROOT / mpath
        exists = full_path.exists()
        assert_check(f"Model 1 file exists: {mpath}", exists, f"File not found: {full_path}")
        if exists:
            try:
                model_obj = joblib.load(full_path)
                assert_check(f"Model 1 object loadable: {mpath}", model_obj is not None)
            except Exception as e:
                assert_check(f"Model 1 object loadable: {mpath}", False, str(e))

    # Model 1 Metadata JSON
    meta_path = PROJECT_ROOT / "user-intent/models/purchase_model_metadata.json"
    assert_check("Model 1 metadata.json exists", meta_path.exists())
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        assert_check("Model 1 metadata contains best_threshold", "best_threshold" in meta)
        assert_check("Model 1 metadata contains metrics", "metrics" in meta and "auc_roc" in meta["metrics"])

    # 2. Model 1 Output CSVs
    prob_path = PROJECT_ROOT / "user-intent/data/session_purchase_probabilities.csv"
    assert_check("Model 1 session_purchase_probabilities.csv exists", prob_path.exists())
    if prob_path.exists():
        p1 = pd.read_csv(prob_path)
        assert_check("Model 1 output has > 1,000 rows", len(p1) >= 1000, f"Got {len(p1)}")
        assert_check("Model 1 has user_session column", "user_session" in p1.columns)
        assert_check("Model 1 has purchase_probability column", "purchase_probability" in p1.columns)
        assert_check("Model 1 has predicted_purchase column", "predicted_purchase" in p1.columns)
        assert_check("Model 1 purchase_probability has no NaNs", p1["purchase_probability"].isna().sum() == 0)
        assert_check("Model 1 purchase_probability range [0, 1]", (p1["purchase_probability"] >= 0).all() and (p1["purchase_probability"] <= 1).all())
        
        # Distribution checks
        prob_std = p1["purchase_probability"].std()
        prob_nunique = p1["purchase_probability"].nunique()
        assert_check("Model 1 purchase probabilities not identical (std > 0.01)", prob_std > 0.01, f"std={prob_std:.4f}")
        assert_check("Model 1 purchase probabilities diverse (> 50 unique)", prob_nunique > 50, f"nunique={prob_nunique}")
        assert_check("Model 1 has both 0 and 1 predictions", set(p1["predicted_purchase"].unique()) == {0, 1})

    # 3. Model 2 Checkpoints Exist & Loadable
    m2_if = PROJECT_ROOT / "behavioral-analysis/models/isolation_forest.pkl"
    m2_scaler = PROJECT_ROOT / "behavioral-analysis/models/feature_scaler.pkl"
    assert_check("Model 2 isolation_forest.pkl exists", m2_if.exists())
    if m2_if.exists():
        try:
            ifo = joblib.load(m2_if)
            assert_check("Model 2 isolation_forest loadable", ifo is not None and hasattr(ifo, "predict"))
        except Exception as e:
            assert_check("Model 2 isolation_forest loadable", False, str(e))

    assert_check("Model 2 feature_scaler.pkl exists", m2_scaler.exists())
    if m2_scaler.exists():
        try:
            scaler = joblib.load(m2_scaler)
            assert_check("Model 2 feature_scaler loadable", scaler is not None and hasattr(scaler, "transform"))
        except Exception as e:
            assert_check("Model 2 feature_scaler loadable", False, str(e))

    # 4. Model 2 Output CSVs
    trust_path = PROJECT_ROOT / "behavioral-analysis/data/processed/final_brand_trust_dataset.csv"
    scores_path = PROJECT_ROOT / "behavioral-analysis/data/output/brand_scores.csv"
    assert_check("Model 2 final_brand_trust_dataset.csv exists", trust_path.exists())
    assert_check("Model 2 brand_scores.csv exists", scores_path.exists())

    if trust_path.exists():
        t2 = pd.read_csv(trust_path)
        assert_check("Model 2 has > 500 brands", len(t2) >= 500, f"Got {len(t2)}")
        for col in ["brand", "suspiciousness_score", "trust_score", "risk_level", "trust_category"]:
            assert_check(f"Model 2 has column: {col}", col in t2.columns)

        assert_check("Model 2 trust_score has no NaNs", t2["trust_score"].isna().sum() == 0)
        assert_check("Model 2 suspiciousness_score has no NaNs", t2["suspiciousness_score"].isna().sum() == 0)
        assert_check("Model 2 trust_score range [0, 1]", (t2["trust_score"] >= 0).all() and (t2["trust_score"] <= 1).all())
        assert_check("Model 2 suspiciousness_score range [0, 1]", (t2["suspiciousness_score"] >= 0).all() and (t2["suspiciousness_score"] <= 1).all())

        # Mathematical consistency: Trust == 1 - Suspiciousness
        diff = np.abs((1.0 - t2["suspiciousness_score"]) - t2["trust_score"]).max()
        assert_check("Model 2 Trust == 1 - Suspiciousness within 1e-6", diff < 1e-6, f"max diff={diff}")

        # Distribution checks
        trust_std = t2["trust_score"].std()
        trust_nunique = t2["trust_score"].nunique()
        assert_check("Model 2 trust scores not identical (std > 0.05)", trust_std > 0.05, f"std={trust_std:.4f}")
        assert_check("Model 2 trust scores diverse (> 50 unique)", trust_nunique > 50, f"nunique={trust_nunique}")

        risk_levels = set(t2["risk_level"].dropna().unique())
        assert_check("Model 2 has multiple risk levels (Low, Medium, High)", {"Low", "Medium", "High"}.issubset(risk_levels), f"Found: {risk_levels}")

    print("\nPhase 1 Summary:")
    print(f"Total Checks: {results['passed'] + results['failed']} | Passed: {results['passed']} | Failed: {results['failed']}")
    return results


if __name__ == "__main__":
    res = test_phase1()
    sys.exit(0 if res["failed"] == 0 else 1)
