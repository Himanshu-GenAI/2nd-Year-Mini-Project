#!/usr/bin/env python3
"""Run complete diagnostic suite on the partial 346 cached weather cells (Day 5 / Week 3).

Executes:
1. KBDI ablation check (G1, G1K, G2, G2R, G3) on 2023 validation primary set.
2. Block bootstrap by calendar week of 2023 (1,000 resamples, seed 42) for PR-AUC differences:
   G2 - G1, G3 - G2, G3 - climatology.
3. Rolling-origin repeat for validation years 2021 and 2022 (climatology, G1, G2, G3).
4. Fire-season view: G1/G2/G3/climatology on peak positive months (February-April)
   derived from 2015-2022 training data.

Outputs saved:
- docs/audit/partial_346_kbdi_ablation.csv
- docs/audit/partial_346_bootstrap.csv
- docs/audit/partial_346_bootstrap_summary.csv
- docs/audit/partial_346_rolling.csv
- docs/audit/partial_346_season.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import DOCS_AUDIT_DIR, FIRE_TABLE_PATH  # noqa: E402
from src.models.diagnostics import (  # noqa: E402
    evaluate_fire_season,
    run_block_bootstrap,
    run_kbdi_ablation,
    run_rolling_origin_eval,
)
from src.models.train_v1 import (  # noqa: E402
    FEATURES_PARTIAL_PATH,
    PARTIAL_CELLS_PATH,
    assert_test_split_locked,
    load_partial_cells,
)


def main() -> int:
    t_start = time.perf_counter()
    assert_test_split_locked()
    print("=" * 78)
    print("RUNNING DIAGNOSTICS ON PARTIAL CACHED CELLS (partial: 346 cells)")
    print("=" * 78)

    cells = load_partial_cells(PARTIAL_CELLS_PATH)
    n_cells = len(cells)
    print(f"Loaded {n_cells} partial cells from {PARTIAL_CELLS_PATH}")
    if n_cells != 346:
        print(f"WARNING: Expected 346 cells, found {n_cells}")

    DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Task 3 & Foundation: KBDI Ablation & 2023 Validation Scoring
    # -------------------------------------------------------------------------
    print("\n[1/4] Running KBDI Ablation & Scoring on 2023 Validation (primary label)...")
    kbdi_df, preds_2023 = run_kbdi_ablation(
        partial_cells=cells,
        features_path=FEATURES_PARTIAL_PATH,
        fire_table_path=FIRE_TABLE_PATH,
        seed=42,
    )
    kbdi_csv = DOCS_AUDIT_DIR / "partial_346_kbdi_ablation.csv"
    kbdi_df.to_csv(kbdi_csv, index=False)
    print(f"Saved KBDI ablation results to {kbdi_csv}")
    print(kbdi_df[["group", "n_features", "pr_auc", "pr_auc_over_base", "recall_top10", "precision_top5", "brier", "best_trees"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # Task 1: Uncertainty Block Bootstrap (1,000 resamples by calendar week of 2023)
    # -------------------------------------------------------------------------
    print("\n[2/4] Running Block Bootstrap (1,000 resamples by calendar week of 2023, seed 42)...")
    score_cols = {
        "climatology": "score_climatology",
        "G1": "score_G1",
        "G2": "score_G2",
        "G3": "score_G3",
    }
    draws_df, boot_summary_df = run_block_bootstrap(
        df=preds_2023,
        label_col="new_fire_next_48h",
        score_cols=score_cols,
        date_col="date",
        n_boot=1000,
        seed=42,
    )
    boot_csv = DOCS_AUDIT_DIR / "partial_346_bootstrap.csv"
    draws_df.to_csv(boot_csv, index=False)
    summary_csv = DOCS_AUDIT_DIR / "partial_346_bootstrap_summary.csv"
    boot_summary_df.to_csv(summary_csv, index=False)
    print(f"Saved bootstrap replicates to {boot_csv}")
    print(f"Saved bootstrap summary to {summary_csv}")
    print(boot_summary_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Task 2: Rolling-Origin Repeat (val years 2021 and 2022)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running Rolling-Origin Repeat for Validation Years 2021 and 2022...")
    rolling_rows = []
    for yr in [2021, 2022]:
        print(f"  Evaluating validation year {yr}...")
        df_yr = run_rolling_origin_eval(
            val_year=yr,
            partial_cells=cells,
            features_path=FEATURES_PARTIAL_PATH,
            fire_table_path=FIRE_TABLE_PATH,
            seed=42,
        )
        rolling_rows.append(df_yr)
    rolling_df = pd.concat(rolling_rows, ignore_index=True)
    rolling_csv = DOCS_AUDIT_DIR / "partial_346_rolling.csv"
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"Saved rolling-origin results to {rolling_csv}")
    print(rolling_df[["val_year", "group", "n_features", "base_rate", "pr_auc", "pr_auc_over_base", "recall_top10", "precision_top5", "brier", "best_trees"]].to_string(index=False))

    # -------------------------------------------------------------------------
    # Task 4: Fire-Season View
    # -------------------------------------------------------------------------
    print("\n[4/4] Evaluating Fire-Season View on Peak Positive Months...")
    # Top 3 peak positive months from 2015-2022 training data: March (6,545), April (5,803), February (5,563)
    peak_3_months = [2, 3, 4]  # Feb, Mar, Apr
    season_3_df = evaluate_fire_season(
        preds_df=preds_2023,
        peak_months=peak_3_months,
        label="new_fire_next_48h",
        set_name=f"partial: {n_cells} cells",
    )
    # Also evaluate pre-monsoon fire season: Feb-May (months 2, 3, 4, 5: 55.9% of positives)
    peak_4_months = [2, 3, 4, 5]
    season_4_df = evaluate_fire_season(
        preds_df=preds_2023,
        peak_months=peak_4_months,
        label="new_fire_next_48h",
        set_name=f"partial: {n_cells} cells",
    )
    combined_season_df = pd.concat([season_3_df, season_4_df], ignore_index=True)
    season_csv = DOCS_AUDIT_DIR / "partial_346_season.csv"
    combined_season_df.to_csv(season_csv, index=False)
    print(f"Saved fire-season results to {season_csv}")
    print(combined_season_df[["season_months", "group", "n_features", "rows", "positives", "base_rate", "pr_auc", "pr_auc_over_base", "recall_top10", "precision_top5", "brier"]].to_string(index=False))

    total_dur = time.perf_counter() - t_start
    print(f"\nAll diagnostics completed successfully in {total_dur:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
