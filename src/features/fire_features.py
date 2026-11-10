"""Fire history and spatial neighbor feature engineering.

Builds 3D spatio-temporal numpy grids counts[date, row, col] and computes
causal rolling windows and Moore-neighborhood fire propagation counts
using purely vectorized array operations.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.config import STUDY_END, STUDY_START, TEST_END, TRAIN_START
from src.data.grid import N_COLS, N_ROWS, load_grid
from src.data.labels import complete_daily_counts


def compute_fire_windows_3d(
    counts: np.ndarray,
    kept_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vectorized 7-day, 30-day, and 3x3 neighbor 7-day fire counts.

    Args:
        counts: 3D array of daily detections shaped (n_dates, n_rows, n_cols).
        kept_mask: 2D boolean mask shaped (n_rows, n_cols) of valid kept cells.

    Returns:
        Tuple of float32 3D arrays: (fire_count_7d, fire_count_30d, neighbour_fire_7d).
    """
    # Zero out any non-kept positions
    masked_counts = np.where(kept_mask[None, :, :], counts, 0).astype(np.int32)

    # Cumulative sum along time axis (axis=0)
    cs = np.cumsum(masked_counts, axis=0)

    # 7-day rolling sum inclusive of t (t-6..t), partial window at start
    s7 = cs.copy()
    if cs.shape[0] > 7:
        s7[7:] = cs[7:] - cs[:-7]

    # 30-day rolling sum inclusive of t (t-29..t), partial window at start
    s30 = cs.copy()
    if cs.shape[0] > 30:
        s30[30:] = cs[30:] - cs[:-30]

    # 3x3 Moore neighborhood sum of 7-day counts minus centre cell
    # Spatial zero-padding ensures borders and non-kept cells contribute 0
    padded_s7 = np.pad(s7, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
    n_r, n_c = counts.shape[1], counts.shape[2]

    sum_3x3 = np.zeros_like(s7, dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            sum_3x3 += padded_s7[:, dr : dr + n_r, dc : dc + n_c]

    neighbour_7d = sum_3x3 - s7

    # Mask again to keep non-kept cells cleanly 0
    neighbour_7d = np.where(kept_mask[None, :, :], neighbour_7d, 0)

    return (
        s7.astype(np.float32),
        s30.astype(np.float32),
        neighbour_7d.astype(np.float32),
    )


def build_fire_features(
    detections_df: pd.DataFrame | None = None,
    grid_df: pd.DataFrame | None = None,
    start_date: str | date = STUDY_START,
    end_date: str | date = STUDY_END,
    eval_start: str | date = TRAIN_START,
    eval_end: str | date = TEST_END,
    n_rows: int = N_ROWS,
    n_cols: int = N_COLS,
) -> pd.DataFrame:
    """Build fire history features for all kept grid cells over the target dates.

    Args:
        detections_df: Clean FIRMS detections. If None, loads from DETECTIONS_CLEAN_PATH.
        grid_df: Grid definition DataFrame. If None, calls load_grid().
        start_date: First date for daily counts array (default 2015-01-01).
        end_date: Last date for daily counts array (default 2025-12-31).
        eval_start: First date to emit in returned DataFrame (default 2015-01-03).
        eval_end: Last date to emit in returned DataFrame (default 2025-12-29).
        n_rows: Grid row dimension (default 64).
        n_cols: Grid column dimension (default 62).

    Returns:
        DataFrame with columns ['cell_id', 'date', 'fire_count_7d', 'fire_count_30d', 'neighbour_fire_7d'].
    """
    if grid_df is None:
        grid_df = load_grid()

    # Build complete counts if default full grid/date run, or compute directly for custom inputs
    if detections_df is None and str(start_date) == STUDY_START and str(end_date) == STUDY_END:
        daily_df = complete_daily_counts(grid_df=grid_df)
        all_dates = pd.date_range(start=STUDY_START, end=STUDY_END, freq="D")
    else:
        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        daily_df = None

    n_dates = len(all_dates)
    counts_3d = np.zeros((n_dates, n_rows, n_cols), dtype=np.uint16)
    kept_mask = np.zeros((n_rows, n_cols), dtype=bool)

    grid_sorted = grid_df.sort_values("cell_id").reset_index(drop=True)
    cell_ids = grid_sorted["cell_id"].astype(str).tolist()
    rows_idx = grid_sorted["row"].astype(int).values
    cols_idx = grid_sorted["col"].astype(int).values

    kept_mask[rows_idx, cols_idx] = True
    cell_to_rc = dict(zip(cell_ids, zip(rows_idx, cols_idx)))
    date_to_t = {d.date(): idx for idx, d in enumerate(all_dates)}

    if daily_df is not None:
        # Map aggregated daily_df directly into 3D tensor
        # Since daily_df is sorted by (cell_id, date) with all n_cells * n_dates rows,
        # we can vectorize mapping via cell row/col indices
        sub_det = daily_df[daily_df["n_det"] > 0]
        if len(sub_det) > 0:
            det_t = sub_det["date"].dt.date.map(date_to_t).values
            det_rc = sub_det["cell_id"].astype(str).map(cell_to_rc)
            det_r = np.array([rc[0] for rc in det_rc], dtype=int)
            det_c = np.array([rc[1] for rc in det_rc], dtype=int)
            counts_3d[det_t, det_r, det_c] = sub_det["n_det"].values
    elif detections_df is not None and len(detections_df) > 0:
        det_dates = pd.to_datetime(detections_df["acq_date"]).dt.date
        det_cells = detections_df["cell_id"].astype(str).values
        for c, d in zip(det_cells, det_dates):
            if c in cell_to_rc and d in date_to_t:
                r, col = cell_to_rc[c]
                t = date_to_t[d]
                counts_3d[t, r, col] += 1

    # Compute 3D window features
    f7_3d, f30_3d, n7_3d = compute_fire_windows_3d(counts_3d, kept_mask)

    # Restrict to eval window (e.g., 2015-01-03 to 2025-12-29)
    eval_s = pd.to_datetime(eval_start)
    eval_e = pd.to_datetime(eval_end)
    date_mask = (all_dates >= eval_s) & (all_dates <= eval_e)
    eval_dates = all_dates[date_mask]
    n_eval_dates = len(eval_dates)

    # Extract kept cell time series in cell-major order (sorted by cell_id, then date)
    # Transpose (T, R, C) -> slice kept cells -> shape (n_eval_dates, n_kept_cells) -> transpose to (n_kept_cells, n_eval_dates)
    f7_kept = f7_3d[date_mask][:, rows_idx, cols_idx].T.flatten()
    f30_kept = f30_3d[date_mask][:, rows_idx, cols_idx].T.flatten()
    n7_kept = n7_3d[date_mask][:, rows_idx, cols_idx].T.flatten()

    cell_col = np.repeat(cell_ids, n_eval_dates)
    date_col = np.tile(eval_dates, len(cell_ids))

    df_out = pd.DataFrame(
        {
            "cell_id": pd.Series(cell_col, dtype="category"),
            "date": pd.Series(date_col, dtype="datetime64[ns]"),
            "fire_count_7d": f7_kept,
            "fire_count_30d": f30_kept,
            "neighbour_fire_7d": n7_kept,
        }
    )
    return df_out
