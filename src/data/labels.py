"""Labels and Fire Cell-Day Panel Generation.

Constructs full spatio-temporal panel (1,167 cells x 4,018 days),
aggregates FIRMS detections, computes backward and forward windows,
generates target classification labels, and maps temporal splits.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FIRE_TABLE_PATH,
    PROJECT_ROOT,
    STUDY_END,
    STUDY_START,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from src.data.firms import load_clean_detections
from src.data.grid import load_grid

logger = logging.getLogger(__name__)

DOCS_AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"


def primary_mask(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for primary evaluation set (not already burning in prev 3 days)."""
    return ~df["burning_prev3"]


def build_full_grid_date_index() -> tuple[list[str], pd.DatetimeIndex]:
    """Generate all cells and date range from STUDY_START to STUDY_END."""
    grid_df = load_grid()
    cells = sorted(grid_df["cell_id"].unique())
    dates = pd.date_range(start=STUDY_START, end=STUDY_END, freq="D")
    return cells, dates


def assign_split(dates: pd.Series) -> pd.Series:
    """Map dates to chronological split: train, val, test, or gap."""
    date_strs = dates.dt.strftime("%Y-%m-%d")

    conditions = [
        (date_strs >= TRAIN_START) & (date_strs <= TRAIN_END),
        (date_strs >= VAL_START) & (date_strs <= VAL_END),
        (date_strs >= TEST_START) & (date_strs <= TEST_END),
    ]
    choices = ["train", "val", "test"]

    split_series = np.select(conditions, choices, default="gap")
    return pd.Series(split_series, index=dates.index, dtype="category")


def build_fire_cell_day_table(
    detections_df: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Construct complete fire cell-day panel with backward/forward windows and labels.

    Args:
        detections_df: DataFrame of clean detections. If None, loads from DETECTIONS_CLEAN_PATH.
        output_path: Path to save Parquet table. If None, uses FIRE_TABLE_PATH.

    Returns:
        Panel DataFrame with 4,684,338 rows (4,014 dates x 1,167 cells).
    """
    out_file = Path(output_path or FIRE_TABLE_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if detections_df is None:
        detections_df = load_clean_detections()

    cells, dates = build_full_grid_date_index()
    n_cells = len(cells)
    n_dates = len(dates)

    cell_to_idx = {c: i for i, c in enumerate(cells)}
    date_to_idx = {d.date(): j for j, d in enumerate(dates)}

    # Initialize 2D matrices for ultra-fast vectorised aggregation
    n_det_mat = np.zeros((n_cells, n_dates), dtype=np.uint16)
    frp_mat = np.zeros((n_cells, n_dates), dtype=np.float32)

    # Aggregate clean detections
    det_dates = pd.to_datetime(detections_df["acq_date"]).dt.date
    det_cells = detections_df["cell_id"].values
    det_frp = detections_df["frp"].values

    for c, d, frp in zip(det_cells, det_dates, det_frp):
        if c in cell_to_idx and d in date_to_idx:
            ci = cell_to_idx[c]
            di = date_to_idx[d]
            n_det_mat[ci, di] += 1
            frp_mat[ci, di] += frp

    # Window calculations for dates t in [2, n_dates - 3] inclusive (indices 2 to 4015)
    # Total valid dates = n_dates - 4 = 4014 dates
    valid_slice = slice(2, n_dates - 2)
    valid_dates = dates[valid_slice]
    n_valid_dates = len(valid_dates)

    # prev3 = sum of n_det over t-2, t-1, t
    prev3_mat = (
        n_det_mat[:, 2 : n_dates - 2]
        + n_det_mat[:, 1 : n_dates - 3]
        + n_det_mat[:, 0 : n_dates - 4]
    )

    # next2 = sum of n_det over t+1, t+2
    next2_mat = (
        n_det_mat[:, 3 : n_dates - 1]
        + n_det_mat[:, 4 : n_dates]
    )

    valid_n_det = n_det_mat[:, valid_slice]
    valid_frp = frp_mat[:, valid_slice]

    # Flags and labels
    burning_prev3_mat = prev3_mat > 0
    fire_next_48h_mat = (next2_mat > 0).astype(np.int8)
    new_fire_next_48h_mat = ((next2_mat > 0) & (~burning_prev3_mat)).astype(np.int8)

    # Flatten into contiguous DataFrame
    # Ordering: cell_id outer, date inner (sorted by cell_id, then date)
    cell_id_col = np.repeat(cells, n_valid_dates)
    date_col = np.tile(valid_dates, n_cells)

    df_panel = pd.DataFrame(
        {
            "cell_id": pd.Series(cell_id_col, dtype="category"),
            "date": pd.Series(date_col, dtype="datetime64[ns]"),
            "n_det": valid_n_det.flatten().astype(np.uint16),
            "frp_sum": valid_frp.flatten().astype(np.float32),
            "burning_prev3": burning_prev3_mat.flatten(),
            "fire_next_48h": fire_next_48h_mat.flatten(),
            "new_fire_next_48h": new_fire_next_48h_mat.flatten(),
        }
    )

    # Split assignment
    df_panel["split"] = assign_split(df_panel["date"])

    # Ensure strictly sorted by cell_id then date
    df_panel = df_panel.sort_values(["cell_id", "date"]).reset_index(drop=True)

    # Save to Parquet
    df_panel.to_parquet(out_file, index=False)
    print(f"Fire cell-day table saved to {out_file} ({len(df_panel):,} rows)")

    # Compute and save label statistics
    compute_and_save_label_stats(df_panel)

    return df_panel


def compute_and_save_label_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute split-level and monthly label statistics and save to CSV."""
    stats_rows = []

    for split_name in ["train", "val", "test", "gap"]:
        sub = df[df["split"] == split_name]
        n_rows = len(sub)
        if n_rows == 0:
            continue

        any_pos = int(sub["fire_next_48h"].sum())
        any_rate = any_pos / n_rows

        burning_pos = int(sub["burning_prev3"].sum())
        burning_share = burning_pos / n_rows

        # Primary set: burning_prev3 == False
        primary_sub = sub[primary_mask(sub)]
        n_primary = len(primary_sub)
        new_pos = int(primary_sub["new_fire_next_48h"].sum())
        new_rate = new_pos / n_primary if n_primary > 0 else 0.0

        stats_rows.append(
            {
                "split": split_name,
                "rows": n_rows,
                "fire_next_48h_positives": any_pos,
                "fire_next_48h_rate": round(any_rate, 6),
                "burning_prev3_share": round(burning_share, 6),
                "primary_rows": n_primary,
                "new_fire_next_48h_positives": new_pos,
                "new_fire_next_48h_rate": round(new_rate, 6),
            }
        )

    # Monthly breakdown for train split
    train_primary = df[(df["split"] == "train") & primary_mask(df)].copy()
    train_primary["month"] = train_primary["date"].dt.month
    monthly_rates = train_primary.groupby("month")["new_fire_next_48h"].mean().to_dict()

    for row in stats_rows:
        if row["split"] == "train":
            for m in range(1, 13):
                row[f"train_m{m:02d}_new_rate"] = round(monthly_rates.get(m, 0.0), 6)

    stats_df_full = pd.DataFrame(stats_rows)
    stats_path = DOCS_AUDIT_DIR / "label_stats.csv"
    stats_df_full.to_csv(stats_path, index=False)
    print(f"Label statistics saved to {stats_path}")
    return stats_df_full
