"""Phase A: Final cell list and weather sanity check for the full-cell run.

Validates all 1,167 cached weather files, computes true daily Tmax extremes,
saves cells_final.json, and reports cells per state.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DOCS_AUDIT_DIR,
    PROCESSED_DIR,
    TMAX_MAX_C,
    TMAX_MIN_C,
    WEATHER_DIR,
    WEATHER_N_DAYS,
)
from src.data.grid import load_grid  # noqa: E402

CELLS_FINAL_PATH = PROCESSED_DIR / "cells_final.json"


def phase_a() -> dict:
    """Run full Phase A validation and produce cells_final.json."""
    t0 = time.perf_counter()

    # 1. Load grid
    grid_df = load_grid()
    cell_ids = sorted(grid_df["cell_id"].astype(str).tolist())
    n_cells = len(cell_ids)
    print(f"Grid loaded: {n_cells} cells")

    # 2. Validate every cell file and collect temperature extremes
    overall_tmax_min = float("inf")
    overall_tmax_max = float("-inf")
    cell_max_tmax = []  # per-cell daily max Tmax
    cell_min_tmax = []  # per-cell daily min Tmax

    missing = []
    failed = []
    complete = []

    expected_dates = pd.date_range("2014-01-01", "2025-12-31", freq="D")

    for cid in cell_ids:
        f_path = WEATHER_DIR / f"{cid}.parquet"
        if not f_path.exists():
            missing.append(cid)
            continue

        try:
            df = pd.read_parquet(f_path)
            n_rows = len(df)

            # Row count
            if n_rows != WEATHER_N_DAYS:
                failed.append((cid, f"Row count {n_rows} != {WEATHER_N_DAYS}"))
                continue

            # Date continuity
            dates_series = pd.to_datetime(df["date"])
            if not (dates_series.dt.date.values == expected_dates.date).all():
                failed.append((cid, "Date continuity mismatch"))
                continue

            # Null check
            n_nulls = int(
                df["temperature_2m_max"].isna().sum()
                + df["precipitation_sum"].isna().sum()
            )
            if n_nulls > 0:
                failed.append((cid, f"{n_nulls} nulls"))
                continue

            tmax_vals = df["temperature_2m_max"].values
            precip_vals = df["precipitation_sum"].values

            # Range check
            if (tmax_vals < TMAX_MIN_C).any() or (tmax_vals > TMAX_MAX_C).any():
                failed.append((cid, f"Tmax out of [{TMAX_MIN_C}, {TMAX_MAX_C}]"))
                continue
            if (precip_vals < 0.0).any():
                failed.append((cid, "Negative precipitation"))
                continue

            # Track extremes
            cell_tmax_max = float(tmax_vals.max())
            cell_tmax_min = float(tmax_vals.min())
            cell_max_tmax.append(cell_tmax_max)
            cell_min_tmax.append(cell_tmax_min)

            if cell_tmax_max > overall_tmax_max:
                overall_tmax_max = cell_tmax_max
            if cell_tmax_min < overall_tmax_min:
                overall_tmax_min = cell_tmax_min

            complete.append(cid)

        except Exception as e:
            failed.append((cid, str(e)))

    n_complete = len(complete)
    n_missing = len(missing)
    n_failed = len(failed)

    print(f"\nWeather validation: {n_complete} complete, {n_missing} missing, {n_failed} failed (total {n_cells})")

    if n_missing > 0:
        print(f"\nSTOP: {n_missing} missing cells:")
        for cid in missing[:20]:
            print(f"  {cid}")
        return {"status": "STOP", "reason": f"{n_missing} missing cells"}

    if n_failed > 0:
        print(f"\nSTOP: {n_failed} failed cells:")
        for cid, reason in failed[:20]:
            print(f"  {cid}: {reason}")
        return {"status": "STOP", "reason": f"{n_failed} failed cells"}

    # 3. Temperature sanity gate
    print("\n--- TEMPERATURE SANITY CHECK (raw daily Tmax) ---")
    print(f"Overall min daily Tmax across all cells: {overall_tmax_min:.2f} °C")
    print(f"Overall max daily Tmax across all cells: {overall_tmax_max:.2f} °C")

    cell_max_arr = np.array(cell_max_tmax)
    p5 = float(np.percentile(cell_max_arr, 5))
    p50 = float(np.percentile(cell_max_arr, 50))
    p95 = float(np.percentile(cell_max_arr, 95))
    print(f"Percentiles of each cell's MAX daily Tmax: p5={p5:.2f}, p50={p50:.2f}, p95={p95:.2f} °C")

    if overall_tmax_max < 40.0:
        print(f"\nSTOP: Overall max daily Tmax ({overall_tmax_max:.2f} °C) < 40 °C — too low for India")
        return {"status": "STOP", "reason": f"Max Tmax {overall_tmax_max:.2f} < 40"}

    if overall_tmax_max > 60.0:
        print(f"\nSTOP: Overall max daily Tmax ({overall_tmax_max:.2f} °C) > 60 °C — unphysical")
        return {"status": "STOP", "reason": f"Max Tmax {overall_tmax_max:.2f} > 60"}

    print("PASS: Temperature range is physically reasonable for India.")

    # 4. Save cells_final.json
    sorted_cells = sorted(complete)
    sha = hashlib.sha256(json.dumps(sorted_cells).encode()).hexdigest()

    cells_final = {
        "count": len(sorted_cells),
        "sha256": sha,
        "cells": sorted_cells,
    }
    CELLS_FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CELLS_FINAL_PATH.write_text(json.dumps(cells_final, indent=2), encoding="utf-8")
    print(f"\nSaved {CELLS_FINAL_PATH} ({len(sorted_cells)} cells, SHA-256: {sha[:16]}...)")

    # 5. Cells per state, states with < 3 cells
    state_map = dict(zip(grid_df["cell_id"].astype(str), grid_df["state"]))
    state_counts: dict[str, int] = {}
    for cid in sorted_cells:
        st = state_map.get(cid, "UNKNOWN")
        state_counts[st] = state_counts.get(st, 0) + 1

    print(f"\n--- CELLS PER STATE ({len(state_counts)} states) ---")
    for st in sorted(state_counts.keys()):
        cnt = state_counts[st]
        flag = " *** < 3 cells ***" if cnt < 3 else ""
        print(f"  {st:30s}: {cnt:4d}{flag}")

    sparse = {st: c for st, c in state_counts.items() if c < 3}
    if sparse:
        print(f"\nStates with < 3 cells ({len(sparse)}):")
        for st, c in sorted(sparse.items()):
            print(f"  {st}: {c}")

    # 6. Write audit CSV with per-cell temperature extremes
    audit_rows = []
    for i, cid in enumerate(sorted_cells):
        audit_rows.append({
            "cell_id": cid,
            "state": state_map.get(cid, "UNKNOWN"),
            "max_daily_tmax": round(cell_max_tmax[i], 2),
            "min_daily_tmax": round(cell_min_tmax[i], 2),
        })
    audit_df = pd.DataFrame(audit_rows)
    audit_path = DOCS_AUDIT_DIR / "cells_final_tmax_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"\nSaved temperature audit to {audit_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nPhase A completed in {elapsed:.1f}s")

    return {
        "status": "PASS",
        "n_cells": len(sorted_cells),
        "sha256": sha,
        "overall_tmax_min": round(overall_tmax_min, 2),
        "overall_tmax_max": round(overall_tmax_max, 2),
        "states": len(state_counts),
        "sparse_states": len(sparse) if sparse else 0,
        "elapsed_s": round(elapsed, 1),
    }


if __name__ == "__main__":
    result = phase_a()
    print(f"\n=== Phase A result: {json.dumps(result, indent=2)} ===")
