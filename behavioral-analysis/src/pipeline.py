# ============================================================
# pipeline.py
# End-to-End Execution Pipeline for Brand Risk & Trust Analysis
# ============================================================

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_PATH,
    FINAL_DATA_PATH,
    MODEL_READY_DATA_PATH,
    MODEL_FEATURE_MATRIX_PATH,
    MODEL_PATH,
    ANOMALY_RESULTS_PATH,
    BRAND_SCORES_PATH,
    TRUST_DATASET_PATH,
    DEFAULT_MAX_CHUNKS,
    CHUNK_SIZE
)
from src.data_loader import check_dataset_exists
from src.category_relative_features import create_category_relative_features
from src.model_preparation import run_phase_7
from src.anomaly_model import run_phase_8
from src.scoring import run_phase9
from src.trust_score import run_phase10


def run_pipeline(force_rerun=False, max_chunks=DEFAULT_MAX_CHUNKS):
    """
    Execute the entire Behavioral Analysis and Brand Trust pipeline:
    - Phase 1-2: Dataset verification & preprocessing
    - Phase 5-6: Category-relative behavioral features (1.5M rows across 3 chunks)
    - Phase 7: Evidence filtering & robust scaling
    - Phase 8: Isolation Forest anomaly training & scoring
    - Phase 9: Brand suspiciousness aggregation
    - Phase 10: Brand trust score calculation
    """
    print("\n" + "=" * 70)
    print("EXECUTING BRAND RISK & TRUST SCORING PIPELINE")
    print("=" * 70)

    # 1. Verify raw clickstream dataset
    print("\n[STEP 1/6] Checking Raw Dataset...")
    check_dataset_exists(RAW_DATA_PATH)

    # 2. Phase 6: Category-Relative Feature Extraction
    if force_rerun or not os.path.exists(FINAL_DATA_PATH):
        print(f"\n[STEP 2/6] Running Phase 6 (Processing {max_chunks} chunks of {CHUNK_SIZE:,} rows)...")
        create_category_relative_features(
            file_path=RAW_DATA_PATH,
            chunk_size=CHUNK_SIZE,
            max_chunks=max_chunks
        )
    else:
        print(f"\n[STEP 2/6] Phase 6 dataset found at:\n  {FINAL_DATA_PATH} (skipping extraction)")

    # 3. Phase 7: Model Preparation & Scaling
    if force_rerun or not os.path.exists(MODEL_READY_DATA_PATH) or not os.path.exists(MODEL_FEATURE_MATRIX_PATH):
        print("\n[STEP 3/6] Running Phase 7: Evidence Filtering & Robust Scaling...")
        run_phase_7()
    else:
        print(f"\n[STEP 3/6] Phase 7 feature matrix found at:\n  {MODEL_FEATURE_MATRIX_PATH} (skipping)")

    # 4. Phase 8: Isolation Forest Anomaly Detection
    if force_rerun or not os.path.exists(MODEL_PATH) or not os.path.exists(ANOMALY_RESULTS_PATH):
        print("\n[STEP 4/6] Running Phase 8: Isolation Forest Training...")
        run_phase_8()
    else:
        print(f"\n[STEP 4/6] Phase 8 trained model found at:\n  {MODEL_PATH} (skipping)")

    # 5. Phase 9: Brand Suspiciousness Aggregation
    if force_rerun or not os.path.exists(BRAND_SCORES_PATH):
        print("\n[STEP 5/6] Running Phase 9: Brand Suspiciousness Aggregation...")
        run_phase9()
    else:
        print(f"\n[STEP 5/6] Phase 9 brand scores found at:\n  {BRAND_SCORES_PATH} (skipping)")

    # 6. Phase 10: Brand Trust Score Calculation
    print("\n[STEP 6/6] Running Phase 10: Brand Trust Score Calculation...")
    run_phase10()

    print("\n" + "=" * 70)
    print("BRAND RISK & TRUST PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Final Trust Dataset: {TRUST_DATASET_PATH}")
    print(f"Brand Scores File:   {BRAND_SCORES_PATH}")
    print(f"Trained Model:       {MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()