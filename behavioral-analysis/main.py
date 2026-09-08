import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline import run_pipeline
from src.trust_score import run_phase10

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRUST-AWARE E-COMMERCE ANALYSIS - BRAND RISK & TRUST PIPELINE")
    print("=" * 70)

    run_pipeline()