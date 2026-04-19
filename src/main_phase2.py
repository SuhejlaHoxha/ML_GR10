"""
main_phase2.py
==============
GitHub Activity-Log — Phase II: Model Training & Analysis
Master's Course – Machine Learning Project

Orchestrates:
  A. Supervised Learning Enhancements & Detailed Analysis
  B. Unsupervised Learning & Clustering Analysis

Run
───
    python src/main_phase2.py

Outputs → phase2_plots/
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from supervised_learning  import run_supervised
from unsupervised_learning import run_unsupervised

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(ROOT_DIR, "processed dataset", "processed_github.csv")
PLOTS_DIR   = os.path.join(ROOT_DIR, "phase2_plots")


def header(title: str) -> None:
    width = 68
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n" + "=" * 68)
    print("  GITHUB ACTIVITY LOG – ML PIPELINE  (Phase II)")
    print(f"  Input  : {CSV_PATH}")
    print(f"  Output : {PLOTS_DIR}/")
    print("=" * 68)

    if not os.path.exists(CSV_PATH):
        print(f"  ✗  Processed CSV not found at:\n     {CSV_PATH}")
        print("  Run Phase I first: python src/main.py")
        return

    # ══════════════════════════════════════════════════════════════════
    # PART A — SUPERVISED LEARNING
    # ══════════════════════════════════════════════════════════════════
    header("PART A · Supervised Learning — 6 Classifiers")
    t0 = time.time()
    try:
        sup_metrics, trained_models, X_test, y_test = run_supervised(
            CSV_PATH, PLOTS_DIR
        )
        print(f"\n  Part A done  ({time.time() - t0:.1f}s)")
    except Exception as exc:
        print(f"  ✗  Supervised learning error: {exc}")
        traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════
    # PART B — UNSUPERVISED LEARNING
    # ══════════════════════════════════════════════════════════════════
    header("PART B · Unsupervised Learning — 4 Clustering Algorithms")
    t0 = time.time()
    try:
        clust_metrics = run_unsupervised(CSV_PATH, PLOTS_DIR)
        print(f"\n  Part B done  ({time.time() - t0:.1f}s)")
    except Exception as exc:
        print(f"  ✗  Unsupervised learning error: {exc}")
        traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    header("PHASE II — COMPLETE")
    plots = sorted([
        f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")
    ])
    print(f"\n  Plots generated : {len(plots)}")
    for p in plots:
        print(f"    {p}")

    csvs = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith(".csv")])
    print(f"\n  CSV reports     : {len(csvs)}")
    for c in csvs:
        print(f"    {c}")

    print(f"\n  All outputs saved to: {PLOTS_DIR}/")
    print("\n  Phase II pipeline complete. ✓\n")


if __name__ == "__main__":
    main()
