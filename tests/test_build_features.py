"""Unit tests for full feature matrix assembly (Day 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    ALL_FEATURE_NAMES,
    BASE_FIRE_FEATURE_NAMES,
    assemble_features_from_components,
    build_synthetic_subset_features,
)
from src.features.registry import all_specs, registry_names


def _make_synthetic_panel(n_cells: int = 4, start: str = "2015-01-03", periods: int = 10):
    cells = [f"r00c{i:02d}" for i in range(n_cells)]
    dates = pd.date_range(start, periods=periods, freq="D")

    rows = []
    for _ci, c in enumerate(cells):
        for d in dates:
            rows.append(
                {
                    "cell_id": c,
                    "date": d,
                    "n_det": 0,
                    "frp_sum": np.float32(0.0),
                    "burning_prev3": False,
                    "new_fire_next_48h": 0,
                    "fire_next_48h": 0,
                    "split": "train",
                }
            )
    fire_table_df = pd.DataFrame(rows)

    cell_coords = {c: (np.float32(20.0 + i * 0.5), np.float32(78.0 + i * 0.5)) for i, c in enumerate(cells)}
    fire_base_df = fire_table_df[["cell_id", "date"]].copy()
    fire_base_df["lat_c"] = fire_base_df["cell_id"].map(lambda x: cell_coords[x][0]).astype(np.float32)
    fire_base_df["lon_c"] = fire_base_df["cell_id"].map(lambda x: cell_coords[x][1]).astype(np.float32)
    fire_base_df["fire_count_7d"] = np.float32(1.0)
    fire_base_df["fire_count_30d"] = np.float32(2.0)
    fire_base_df["neighbour_fire_7d"] = np.float32(0.5)
    doy_month = pd.to_datetime(fire_base_df["date"]).dt.month
    fire_base_df["month_sin"] = np.sin(2.0 * np.pi * doy_month / 12.0).astype(np.float32)
    fire_base_df["month_cos"] = np.cos(2.0 * np.pi * doy_month / 12.0).astype(np.float32)
    fire_base_df = fire_base_df[["cell_id", "date", *BASE_FIRE_FEATURE_NAMES]]

    weather_cols = [c for c in ALL_FEATURE_NAMES if c not in BASE_FIRE_FEATURE_NAMES]
    weather_df = fire_table_df[["cell_id", "date"]].copy()
    for j, col in enumerate(weather_cols):
        weather_df[col] = np.float32(10.0 + j)

    return fire_table_df, fire_base_df, weather_df


def test_registry_completeness_21_features():
    names = registry_names()
    assert names == ALL_FEATURE_NAMES
    assert len(ALL_FEATURE_NAMES) == 21
    assert len(BASE_FIRE_FEATURE_NAMES) == 7
    assert len(set(ALL_FEATURE_NAMES)) == 21
    for spec in all_specs():
        assert spec.uses_future is False
        assert spec.dtype == "float32"


def test_assemble_features_full_and_partial():
    fire_table_df, fire_base_df, weather_df = _make_synthetic_panel(n_cells=4, periods=10)

    full_df = assemble_features_from_components(
        fire_table_df=fire_table_df,
        fire_base_df=fire_base_df,
        weather_df=weather_df,
        partial=False,
    )
    assert len(full_df) == 40
    assert not full_df.duplicated(subset=["cell_id", "date"]).any()
    for col in ALL_FEATURE_NAMES:
        assert col in full_df.columns
        assert full_df[col].dtype == np.float32
        assert not full_df[col].isna().any()
        assert np.isfinite(full_df[col].values).all()

    # Partial mode with only 2 of 4 cells having weather
    partial_weather = weather_df[weather_df["cell_id"].isin(["r00c00", "r00c01"])].copy()
    partial_df = assemble_features_from_components(
        fire_table_df=fire_table_df,
        fire_base_df=fire_base_df,
        weather_df=partial_weather,
        partial=True,
    )
    assert len(partial_df) == 20
    assert "has_weather" in partial_df.columns
    assert bool(partial_df["has_weather"].all())
    assert set(partial_df["cell_id"].unique()) == {"r00c00", "r00c01"}
    for col in ALL_FEATURE_NAMES:
        assert partial_df[col].dtype == np.float32
        assert not partial_df[col].isna().any()
        assert np.isfinite(partial_df[col].values).all()

    # Full mode must refuse missing weather cells instead of leaving NaNs
    with pytest.raises((ValueError, AssertionError, KeyError)):
        assemble_features_from_components(
            fire_table_df=fire_table_df,
            fire_base_df=fire_base_df,
            weather_df=partial_weather,
            partial=False,
        )


def test_build_synthetic_subset_features_and_dry_vs_monsoon():
    grid_df = pd.DataFrame(
        {
            "cell_id": ["r00c00", "r00c01", "r01c00", "r01c01"],
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "lat_c": [20.25, 20.25, 20.75, 20.75],
            "lon_c": [78.25, 78.75, 78.25, 78.75],
        }
    )
    weather_dates = pd.date_range("2014-12-01", "2015-03-31", freq="D")
    n_days = len(weather_dates)
    n_cells = len(grid_df)

    # First ~75 days dry & hot; remaining days cool & rainy (monsoon)
    tmax_c = np.full((n_days, n_cells), 38.0, dtype=np.float32)
    precip_mm = np.zeros((n_days, n_cells), dtype=np.float32)
    wet_mask = weather_dates >= pd.Timestamp("2015-02-15")
    tmax_c[wet_mask, :] = 24.0
    precip_mm[wet_mask, :] = 25.0
    mean_annual_mm = np.full(n_cells, 1000.0, dtype=np.float32)

    detections_df = pd.DataFrame(
        {
            "cell_id": ["r00c00", "r01c01"],
            "acq_date": [pd.Timestamp("2015-01-10").date(), pd.Timestamp("2015-01-15").date()],
        }
    )

    eval_dates = pd.date_range("2015-01-03", "2015-03-31", freq="D")
    ft_rows = []
    for c in grid_df["cell_id"]:
        for d in eval_dates:
            ft_rows.append(
                {
                    "cell_id": c,
                    "date": d,
                    "n_det": 0,
                    "frp_sum": np.float32(0.0),
                    "burning_prev3": False,
                    "new_fire_next_48h": 0,
                    "fire_next_48h": 0,
                    "split": "train",
                }
            )
    fire_table_df = pd.DataFrame(ft_rows)

    feat_df = build_synthetic_subset_features(
        detections_df=detections_df,
        grid_df=grid_df,
        tmax_c=tmax_c,
        precip_mm=precip_mm,
        weather_dates=weather_dates,
        mean_annual_mm=mean_annual_mm,
        fire_table_df=fire_table_df,
        start_date="2015-01-01",
        end_date="2015-03-31",
        eval_start="2015-01-03",
        eval_end="2015-03-31",
        n_rows=2,
        n_cols=2,
    )

    assert len(feat_df) == len(fire_table_df)
    assert not feat_df.duplicated(subset=["cell_id", "date"]).any()
    for col in ALL_FEATURE_NAMES:
        assert feat_df[col].dtype == np.float32
        assert not feat_df[col].isna().any()
        assert np.isfinite(feat_df[col].values).all()

    dry_slice = feat_df[feat_df["date"] == pd.Timestamp("2015-02-10")]
    wet_slice = feat_df[feat_df["date"] == pd.Timestamp("2015-03-10")]
    assert dry_slice["kbdi"].mean() > wet_slice["kbdi"].mean()
    assert dry_slice["tmax_mean_7d"].mean() > wet_slice["tmax_mean_7d"].mean()
    assert dry_slice["days_since_rain"].mean() > wet_slice["days_since_rain"].mean()
