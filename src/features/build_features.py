"""Full and Partial Feature Matrix Builder (Day 4).

Assembles fire history, calendar, static spatial attributes, meteorological lags,
and KBDI drought features into a unified leakage-tested training table.
Supports both full 1,167-cell mode (features.parquet) and partial-cell mode
(features_partial.parquet + partial_cells.json).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import FIRE_TABLE_PATH, PROCESSED_DIR
from src.data.grid import load_grid
from src.features.base_features import FEATURES_FIRE_BASE_PATH
from src.features.fire_features import build_fire_features
from src.features.registry import registry_names
from src.features.weather_features import (
    MEAN_ANNUAL_RAIN_PATH,
    WEATHER_FEATURE_NAMES,
    compute_weather_and_drought_features,
    load_weather_matrix,
    mean_annual_rainfall_mm,
)

FEATURES_FULL_PATH = PROCESSED_DIR / "features.parquet"
FEATURES_PARTIAL_PATH = PROCESSED_DIR / "features_partial.parquet"
PARTIAL_CELLS_JSON_PATH = PROCESSED_DIR / "partial_cells.json"

ALL_FEATURE_NAMES = registry_names()
BASE_FIRE_FEATURE_NAMES = [
    "fire_count_7d",
    "fire_count_30d",
    "neighbour_fire_7d",
    "month_sin",
    "month_cos",
    "lat_c",
    "lon_c",
]


def assemble_features_from_components(
    fire_table_df: pd.DataFrame,
    fire_base_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    partial: bool = False,
) -> pd.DataFrame:
    """Merge fire table labels, fire/base features, and weather/KBDI features cleanly.

    Args:
        fire_table_df: DataFrame containing keys and labels (cell_id, date, split, labels).
        fire_base_df: DataFrame containing fire history, calendar, and static features.
        weather_df: DataFrame containing the 14 weather and KBDI features.
        partial: If True, inner-joins to cells present in weather_df and sets has_weather=True.

    Returns:
        Sorted, validated feature panel DataFrame with zero NaNs/Infs and float32 features.
    """
    ft = fire_table_df.copy()
    fb = fire_base_df.copy()
    wd = weather_df.copy()

    ft["cell_id"] = ft["cell_id"].astype(str)
    fb["cell_id"] = fb["cell_id"].astype(str)
    wd["cell_id"] = wd["cell_id"].astype(str)

    if partial:
        valid_cells = set(wd["cell_id"].unique())
        ft = ft[ft["cell_id"].isin(valid_cells)].copy()
        fb = fb[fb["cell_id"].isin(valid_cells)].copy()

    # Merge fire/base onto fire table
    merged = ft.merge(
        fb[["cell_id", "date"] + BASE_FIRE_FEATURE_NAMES],
        on=["cell_id", "date"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(ft):
        raise ValueError(f"Fire+base merge row count mismatch: expected {len(ft)}, got {len(merged)}")

    # Merge weather/KBDI features
    merged = merged.merge(
        wd[["cell_id", "date"] + WEATHER_FEATURE_NAMES],
        on=["cell_id", "date"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(ft):
        raise ValueError(f"Weather merge row count mismatch: expected {len(ft)}, got {len(merged)}")

    if partial:
        merged["has_weather"] = True

    # Validate zero NaNs, zero Infs, and float32 dtypes
    for col in ALL_FEATURE_NAMES:
        merged[col] = merged[col].astype(np.float32)
        if merged[col].isna().any():
            raise ValueError(f"Feature '{col}' contains NaNs")
        if np.isinf(merged[col].values).any():
            raise ValueError(f"Feature '{col}' contains Inf values")

    # Validate unique (cell_id, date)
    if merged.duplicated(subset=["cell_id", "date"]).any():
        raise ValueError("Duplicate (cell_id, date) keys detected in assembled features table")

    merged["cell_id"] = merged["cell_id"].astype("category")
    merged["split"] = merged["split"].astype("category")
    merged = merged.sort_values(["cell_id", "date"]).reset_index(drop=True)

    return merged


def build_features_table(
    partial: bool = False,
    output_path: Path | str | None = None,
    fire_table_path: Path | str | None = None,
    fire_base_path: Path | str | None = None,
    weather_dir: Path | str | None = None,
    grid_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build and write full (or partial) feature matrix Parquet with zstd compression."""
    t0 = time.perf_counter()

    if grid_df is None:
        grid_df = load_grid()

    tmax_all, precip_all, dates, cell_ids, has_weather = load_weather_matrix(
        weather_dir=weather_dir,
        grid_df=grid_df,
    )
    valid_idx = np.where(has_weather)[0]
    valid_cells = [cell_ids[i] for i in valid_idx]

    if not partial and len(valid_cells) != len(cell_ids):
        raise RuntimeError(
            f"Cannot build full features.parquet: only {len(valid_cells)}/{len(cell_ids)} "
            f"weather cells complete. Run with partial=True after >=300 cells or wait for full fetch."
        )

    if partial and len(valid_cells) == 0:
        raise RuntimeError("No valid cached weather cells found for partial build.")

    # Compute mean annual rainfall (2015-2022) and weather/KBDI features
    tmax_valid = tmax_all[:, valid_idx]
    precip_valid = precip_all[:, valid_idx]

    mean_rain = mean_annual_rainfall_mm(
        precip_valid,
        dates,
        cell_ids=valid_cells,
        save_path=MEAN_ANNUAL_RAIN_PATH if not partial else None,
    )
    weather_df = compute_weather_and_drought_features(
        tmax_c=tmax_valid,
        precip_mm=precip_valid,
        mean_annual_mm=mean_rain,
        dates=dates,
        cell_ids=valid_cells,
    )

    ft_path = Path(fire_table_path or FIRE_TABLE_PATH)
    fb_path = Path(fire_base_path or FEATURES_FIRE_BASE_PATH)

    fire_table_df = pd.read_parquet(ft_path)
    fire_base_df = pd.read_parquet(fb_path)

    full_df = assemble_features_from_components(
        fire_table_df=fire_table_df,
        fire_base_df=fire_base_df,
        weather_df=weather_df,
        partial=partial,
    )

    if not partial:
        # Strict full-table row and split count verification
        assert len(full_df) == 4684338, f"Expected 4,684,338 rows, got {len(full_df)}"
        split_counts = full_df["split"].value_counts().to_dict()
        assert split_counts["train"] == 3405306
        assert split_counts["val"] == 423621
        assert split_counts["test"] == 850743
        assert split_counts["gap"] == 4668
        default_out = FEATURES_FULL_PATH
    else:
        assert len(full_df) == len(valid_cells) * 4014
        PARTIAL_CELLS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PARTIAL_CELLS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(valid_cells), f, indent=2)
        default_out = FEATURES_PARTIAL_PATH

    out_file = Path(output_path or default_out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_file, index=False, compression="zstd")

    elapsed = time.perf_counter() - t0
    size_mb = out_file.stat().st_size / (1024 * 1024)
    meta = {
        "rows": len(full_df),
        "n_cells": len(valid_cells),
        "partial": partial,
        "output_path": str(out_file),
        "size_mb": round(size_mb, 2),
        "seconds": round(elapsed, 2),
    }
    print(f"Built features table ({len(full_df):,} rows, {len(valid_cells)} cells, {size_mb:.2f} MB) in {elapsed:.1f}s")
    return full_df, meta


def build_synthetic_subset_features(
    detections_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    tmax_c: np.ndarray,
    precip_mm: np.ndarray,
    weather_dates: pd.DatetimeIndex,
    mean_annual_mm: np.ndarray,
    fire_table_df: pd.DataFrame,
    start_date: str = "2015-01-01",
    end_date: str = "2015-04-01",
    eval_start: str = "2015-01-03",
    eval_end: str = "2015-04-01",
    n_rows: int = 10,
    n_cols: int = 10,
) -> pd.DataFrame:
    """In-memory full-pipeline feature builder for synthetic leakage and integration tests."""
    from src.features.base_features import build_base_features

    cell_ids = sorted(grid_df["cell_id"].astype(str).tolist())
    fire_df = build_fire_features(
        detections_df=detections_df,
        grid_df=grid_df,
        start_date=start_date,
        end_date=end_date,
        eval_start=eval_start,
        eval_end=eval_end,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    eval_dates = pd.date_range(start=eval_start, end=eval_end, freq="D")
    base_df = build_base_features(grid_df, eval_dates)
    fb_df = fire_df.copy()
    for col in ["month_sin", "month_cos", "lat_c", "lon_c"]:
        fb_df[col] = base_df[col].values

    wd_df = compute_weather_and_drought_features(
        tmax_c=tmax_c,
        precip_mm=precip_mm,
        mean_annual_mm=mean_annual_mm,
        dates=weather_dates,
        cell_ids=cell_ids,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    return assemble_features_from_components(fire_table_df=fire_table_df, fire_base_df=fb_df, weather_df=wd_df)
