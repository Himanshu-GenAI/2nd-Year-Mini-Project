"""Spatial 0.5-degree grid mesh utilities for wildfire discretization.

Calculates cell row/column coordinates, generates standardized cell identifiers,
and loads validated spatial grid geometries for the India study domain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    GRID_DEG,
    GRID_PATH,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
)

N_ROWS = int(round((LAT_MAX - LAT_MIN) / GRID_DEG))  # 64
N_COLS = int(round((LON_MAX - LON_MIN) / GRID_DEG))  # 62


def cell_row_col(
    lat: float | np.ndarray | pd.Series,
    lon: float | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized calculation of spatial grid row and column indices.

    Points exactly on LAT_MAX or LON_MAX are considered outside the grid.
    Points outside [LAT_MIN, LAT_MAX) or [LON_MIN, LON_MAX) return -1.

    Args:
        lat: Latitude(s) in decimal degrees.
        lon: Longitude(s) in decimal degrees.

    Returns:
        Tuple of (rows, cols) as int64 numpy arrays, with -1 denoting out-of-bounds.
    """
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)

    is_scalar = lat_arr.ndim == 0 and lon_arr.ndim == 0
    if is_scalar:
        lat_arr = np.atleast_1d(lat_arr)
        lon_arr = np.atleast_1d(lon_arr)

    rows = np.floor((lat_arr - LAT_MIN) / GRID_DEG).astype(np.int64)
    cols = np.floor((lon_arr - LON_MIN) / GRID_DEG).astype(np.int64)

    # Upper boundary is strictly non-inclusive: lat == LAT_MAX or lon == LON_MAX is outside
    valid_mask = (
        (lat_arr >= LAT_MIN)
        & (lat_arr < LAT_MAX)
        & (lon_arr >= LON_MIN)
        & (lon_arr < LON_MAX)
        & (rows >= 0)
        & (rows < N_ROWS)
        & (cols >= 0)
        & (cols < N_COLS)
    )

    rows = np.where(valid_mask, rows, -1)
    cols = np.where(valid_mask, cols, -1)

    if is_scalar:
        return rows[0], cols[0]
    return rows, cols


def make_cell_id(
    row: int | np.ndarray | pd.Series,
    col: int | np.ndarray | pd.Series,
) -> str | np.ndarray:
    """Generate standardized cell identifier 'r{row:02d}c{col:02d}'.

    Returns empty string for out-of-bounds index (-1).
    """
    row_arr = np.asarray(row)
    col_arr = np.asarray(col)

    is_scalar = row_arr.ndim == 0 and col_arr.ndim == 0
    if is_scalar:
        r, c = int(row_arr), int(col_arr)
        if r < 0 or c < 0:
            return ""
        return f"r{r:02d}c{c:02d}"

    row_flat = row_arr.flat
    col_flat = col_arr.flat
    result = np.array(
        [f"r{int(r):02d}c{int(c):02d}" if (r >= 0 and c >= 0) else "" for r, c in zip(row_flat, col_flat)],
        dtype=object,
    ).reshape(row_arr.shape)
    return result


def load_grid(grid_path: str | pd.DataFrame | None = None) -> pd.DataFrame:
    """Load and validate national 0.5-degree grid cells dataset.

    Validates:
      - Exactly 1,167 active grid cells
      - Unique cell_id column
      - Required coordinate & boundary columns present
      - Each cell_id matches make_cell_id(row, col)

    Returns:
        Validated grid DataFrame.
    """
    path = grid_path or GRID_PATH
    df = pd.read_parquet(path)

    expected_rows = 1167
    if len(df) != expected_rows:
        raise ValueError(f"Grid row count mismatch. Expected {expected_rows}, got {len(df)}")

    required_cols = {
        "cell_id",
        "row",
        "col",
        "lat_c",
        "lon_c",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "inside_share",
        "state",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Grid missing required columns: {missing}")

    if df["cell_id"].duplicated().any():
        raise ValueError("Duplicate cell_id detected in grid file")

    expected_ids = [f"r{int(r):02d}c{int(c):02d}" for r, c in zip(df["row"], df["col"])]
    if (df["cell_id"] != expected_ids).any():
        raise ValueError("Cell ID format does not match make_cell_id(row, col)")

    return df
