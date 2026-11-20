"""Phase B: Build full features.parquet for all 1,167 cells.

Builds the complete feature table, verifies row counts and splits,
runs seasonal sanity checks.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import PROCESSED_DIR  # noqa: E402
from src.features.build_features import FEATURES_FULL_PATH, build_features_table  # noqa: E402


def phase_b() -> dict:
    """Build full features.parquet and run sanity checks."""
    tracemalloc.start()
    t0 = time.perf_counter()

    # 1. Load cells_final.json to verify cell count
    cells_final_path = PROCESSED_DIR / "cells_final.json"
    with open(cells_final_path, encoding="utf-8") as f:
        cells_final = json.load(f)
    expected_cells = cells_final["count"]
    print(f"cells_final.json: {expected_cells} cells, SHA-256: {cells_final['sha256'][:16]}...")

    # 2. Build full features table (partial=False)
    print(f"\nBuilding full features.parquet for {expected_cells} cells...")
    full_df, meta = build_features_table(partial=False)

    build_time = time.perf_counter() - t0
    peak_mem_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)

    print(f"\nBuild complete: {meta['rows']:,} rows, {meta['n_cells']} cells, "
          f"{meta['size_mb']:.2f} MB, {build_time:.1f}s")
    print(f"Peak memory (tracemalloc): {peak_mem_mb:.1f} MB")

    # 3. Verify row counts
    expected_rows = expected_cells * 4014
    actual_rows = len(full_df)
    print(f"\nRow count: expected {expected_rows:,}, actual {actual_rows:,}")
    assert actual_rows == 4684338, f"Row count mismatch: {actual_rows} != 4,684,338"

    # 4. Verify split counts
    split_counts = full_df["split"].value_counts().to_dict()
    print("\nSplit counts:")
    for split_name in ["train", "val", "test", "gap"]:
        cnt = split_counts.get(split_name, 0)
        print(f"  {split_name}: {cnt:,}")

    assert split_counts["train"] == 3405306, f"Train count: {split_counts['train']}"
    assert split_counts["val"] == 423621, f"Val count: {split_counts['val']}"
    assert split_counts["test"] == 850743, f"Test count: {split_counts['test']}"
    assert split_counts["gap"] == 4668, f"Gap count: {split_counts['gap']}"
    print("All split counts match expected values.")

    # 5. Zero NaN check
    n_nan = int(full_df.isna().sum().sum())
    print(f"\nNaN count across all columns: {n_nan}")
    assert n_nan == 0, f"Found {n_nan} NaN values!"

    # 6. Seasonal sanity checks
    print("\n--- SEASONAL SANITY CHECKS ---")
    full_df["month"] = pd.to_datetime(full_df["date"]).dt.month
    train_val = full_df[full_df["split"].isin(["train", "val"])].copy()

    # KBDI: higher in pre-monsoon (Mar-May) vs monsoon (Jul-Sep)
    pre_monsoon = train_val[train_val["month"].isin([3, 4, 5])]
    monsoon = train_val[train_val["month"].isin([7, 8, 9])]
    kbdi_premonsoon = float(pre_monsoon["kbdi"].mean())
    kbdi_monsoon = float(monsoon["kbdi"].mean())
    print(f"Mean KBDI: pre-monsoon (Mar-May) = {kbdi_premonsoon:.2f}, monsoon (Jul-Sep) = {kbdi_monsoon:.2f}")
    assert kbdi_premonsoon > kbdi_monsoon, "KBDI should be higher in pre-monsoon than monsoon"
    print("  PASS: KBDI is higher in pre-monsoon as expected.")

    # precip_sum_30d: higher in monsoon (Jul-Sep) vs pre-monsoon (Mar-May)
    precip30_premonsoon = float(pre_monsoon["precip_sum_30d"].mean())
    precip30_monsoon = float(monsoon["precip_sum_30d"].mean())
    print(f"Mean precip_sum_30d: pre-monsoon = {precip30_premonsoon:.2f}, monsoon = {precip30_monsoon:.2f}")
    assert precip30_monsoon > precip30_premonsoon, "precip_sum_30d should be higher in monsoon"
    print("  PASS: 30-day precip is higher in monsoon as expected.")

    # 7. Feature column check
    from src.features.build_features import ALL_FEATURE_NAMES
    missing_feats = [f for f in ALL_FEATURE_NAMES if f not in full_df.columns]
    print(f"\nFeature columns: {len(ALL_FEATURE_NAMES)} expected, {len(missing_feats)} missing")
    assert len(missing_feats) == 0, f"Missing features: {missing_feats}"
    print("All feature columns present.")

    # 8. Summary stats for key features
    print("\n--- KEY FEATURE STATISTICS (train+val) ---")
    for col in ["kbdi", "tmax_t0", "precip_t0", "precip_sum_30d", "fire_count_7d"]:
        vals = train_val[col]
        print(f"  {col:20s}: mean={float(vals.mean()):.4f}, std={float(vals.std()):.4f}, "
              f"min={float(vals.min()):.4f}, max={float(vals.max()):.4f}")

    elapsed = time.perf_counter() - t0
    tracemalloc.stop()

    result = {
        "status": "PASS",
        "rows": actual_rows,
        "n_cells": meta["n_cells"],
        "size_mb": meta["size_mb"],
        "build_time_s": round(build_time, 1),
        "peak_memory_mb": round(peak_mem_mb, 1),
        "output_path": str(FEATURES_FULL_PATH),
    }
    print(f"\nPhase B completed in {elapsed:.1f}s")
    print(f"Output: {FEATURES_FULL_PATH}")
    return result


if __name__ == "__main__":
    result = phase_b()
    print(f"\n=== Phase B result: {json.dumps(result, indent=2)} ===")
