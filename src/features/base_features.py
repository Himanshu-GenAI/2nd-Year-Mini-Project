"""Calendar and static geographic features builder.

Generates cyclical seasonal encodings (month_sin, month_cos) and static cell
coordinates (lat_c, lon_c), and assembles the non-weather intermediate feature table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TEST_END, TRAIN_START
from src.data.grid import load_grid
from src.features.fire_features import build_fire_features

FEATURES_FIRE_BASE_PATH = PROCESSED_DIR / "features_fire_base.parquet"


def build_base_features(
    grid_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build calendar (month_sin, month_cos) and static (lat_c, lon_c) features.

    Args:
        grid_df: Grid DataFrame containing ['cell_id', 'lat_c', 'lon_c'].
        dates: DatetimeIndex of evaluation dates.

    Returns:
        DataFrame with columns ['cell_id', 'date', 'month_sin', 'month_cos', 'lat_c', 'lon_c'].
    """
    grid_sorted = grid_df.sort_values("cell_id").reset_index(drop=True)
    n_cells = len(grid_sorted)
    n_dates = len(dates)

    months = dates.month.values
    m_sin_1d = np.sin(2.0 * np.pi * months / 12.0).astype(np.float32)
    m_cos_1d = np.cos(2.0 * np.pi * months / 12.0).astype(np.float32)

    cell_col = np.repeat(grid_sorted["cell_id"].astype(str).values, n_dates)
    date_col = np.tile(dates, n_cells)
    month_sin_col = np.tile(m_sin_1d, n_cells)
    month_cos_col = np.tile(m_cos_1d, n_cells)
    lat_col = np.repeat(grid_sorted["lat_c"].values.astype(np.float32), n_dates)
    lon_col = np.repeat(grid_sorted["lon_c"].values.astype(np.float32), n_dates)

    return pd.DataFrame(
        {
            "cell_id": pd.Series(cell_col, dtype="category"),
            "date": pd.Series(date_col, dtype="datetime64[ns]"),
            "month_sin": month_sin_col,
            "month_cos": month_cos_col,
            "lat_c": lat_col,
            "lon_c": lon_col,
        }
    )


def build_and_save_fire_base_features(
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Assemble and save fire, calendar, and static features (4,684,338 rows)."""
    out_file = Path(output_path or FEATURES_FIRE_BASE_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    grid_df = load_grid()
    fire_df = build_fire_features(grid_df=grid_df)

    eval_dates = pd.date_range(start=TRAIN_START, end=TEST_END, freq="D")
    base_df = build_base_features(grid_df, eval_dates)

    # Since both are strictly sorted by (cell_id, date) over identical index, validate alignment and combine
    assert len(fire_df) == len(base_df), f"Row count mismatch: {len(fire_df)} vs {len(base_df)}"
    assert (fire_df["cell_id"].values == base_df["cell_id"].values).all(), "cell_id alignment mismatch"
    assert (fire_df["date"].values == base_df["date"].values).all(), "date alignment mismatch"

    merged = fire_df.copy()
    for col in ["month_sin", "month_cos", "lat_c", "lon_c"]:
        merged[col] = base_df[col].values

    # Check zero NaNs and float32 dtypes
    feature_cols = [
        "fire_count_7d",
        "fire_count_30d",
        "neighbour_fire_7d",
        "month_sin",
        "month_cos",
        "lat_c",
        "lon_c",
    ]
    for col in feature_cols:
        assert merged[col].dtype == np.float32, f"Expected float32 for {col}, got {merged[col].dtype}"
        assert merged[col].isna().sum() == 0, f"Found NaNs in {col}"

    merged.to_parquet(out_file, index=False)
    print(f"Saved fire+base features to {out_file} ({len(merged):,} rows)")
    return merged
