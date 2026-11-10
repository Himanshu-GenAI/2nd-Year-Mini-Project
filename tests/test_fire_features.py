from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import FIRE_TABLE_PATH
from src.features.base_features import build_base_features
from src.features.fire_features import build_fire_features
from src.features.registry import all_specs, get, registry_names


def _make_3cell_grid() -> pd.DataFrame:
    # 3 cells in a row: (row=0, col=0), (row=0, col=1), (row=0, col=2)
    # Note: cell r00c01 has two kept neighbors (r00c00 and r00c02).
    # cell r00c00 is on edge: has only one kept neighbor (r00c01).
    return pd.DataFrame(
        {
            "cell_id": ["r00c00", "r00c01", "r00c02"],
            "row": [0, 0, 0],
            "col": [0, 1, 2],
            "lat_c": [6.25, 6.25, 6.25],
            "lon_c": [67.25, 67.75, 68.25],
        }
    )


def test_toy_3cells_hand_computed_windows():
    grid_df = _make_3cell_grid()
    # 40 days starting 2015-01-01 to 2015-02-09
    # Detections:
    # r00c00: 2 dets on 2015-01-01, 1 det on 2015-01-03, 1 det on 2015-01-10
    # r00c01: 1 det on 2015-01-05
    # r00c02: 3 dets on 2015-01-06
    detections = pd.DataFrame(
        {
            "cell_id": ["r00c00", "r00c00", "r00c00", "r00c00", "r00c01", "r00c02", "r00c02", "r00c02"],
            "acq_date": [
                date(2015, 1, 1),
                date(2015, 1, 1),
                date(2015, 1, 3),
                date(2015, 1, 10),
                date(2015, 1, 5),
                date(2015, 1, 6),
                date(2015, 1, 6),
                date(2015, 1, 6),
            ],
        }
    )

    res = build_fire_features(
        detections_df=detections,
        grid_df=grid_df,
        start_date="2015-01-01",
        end_date="2015-02-09",
        eval_start="2015-01-03",
        eval_end="2015-02-09",
        n_rows=3,
        n_cols=4,
    )

    # Check partial window on 2015-01-03 for r00c00: sum of 2015-01-01..2015-01-03 = 2 + 1 = 3.0
    row_jan3 = res[(res["cell_id"] == "r00c00") & (res["date"] == pd.Timestamp("2015-01-03"))].iloc[0]
    assert row_jan3["fire_count_7d"] == 3.0
    assert row_jan3["fire_count_30d"] == 3.0
    assert row_jan3["neighbour_fire_7d"] == 0.0

    # Check 2015-01-08 for r00c01 (middle cell):
    # r00c01 own 7d (2015-01-02..2015-01-08): 1 det (Jan 5) -> 1.0
    # r00c01 30d: 1.0
    # Neighbors in 7d (Jan 2..Jan 8): r00c00 has 1 (Jan 3), r00c02 has 3 (Jan 6) -> 4.0
    row_c1_jan8 = res[(res["cell_id"] == "r00c01") & (res["date"] == pd.Timestamp("2015-01-08"))].iloc[0]
    assert row_c1_jan8["fire_count_7d"] == 1.0
    assert row_c1_jan8["fire_count_30d"] == 1.0
    assert row_c1_jan8["neighbour_fire_7d"] == 4.0

    # Edge cell r00c00 on 2015-01-10:
    # r00c00 own 7d (Jan 4..Jan 10): 1 det (Jan 10) -> 1.0
    # r00c00 own 30d: 2 (Jan 1) + 1 (Jan 3) + 1 (Jan 10) = 4.0
    # Neighbor of r00c00 is only r00c01 (Jan 4..Jan 10 has 1 det on Jan 5). r00c02 is 2 cols away, not neighbor!
    row_c0_jan10 = res[(res["cell_id"] == "r00c00") & (res["date"] == pd.Timestamp("2015-01-10"))].iloc[0]
    assert row_c0_jan10["fire_count_7d"] == 1.0
    assert row_c0_jan10["fire_count_30d"] == 4.0
    assert row_c0_jan10["neighbour_fire_7d"] == 1.0

    # On 2015-02-05 (day 36): Jan 1 detections (index 0) fall out of 30d window (Jan 7..Feb 5)!
    # Jan 10 detection (1) stays in 30d window. Jan 3 (index 2) also fell out!
    row_c0_feb5 = res[(res["cell_id"] == "r00c00") & (res["date"] == pd.Timestamp("2015-02-05"))].iloc[0]
    assert row_c0_feb5["fire_count_7d"] == 0.0
    assert row_c0_feb5["fire_count_30d"] == 1.0


def test_month_cyclical_encoding_distance():
    grid_df = _make_3cell_grid()
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="MS")
    base_df = build_base_features(grid_df.iloc[:1], dates)

    pts = base_df[["month_sin", "month_cos"]].values
    # Distance between Dec (idx 11) and Jan (idx 0)
    dist_dec_jan = np.linalg.norm(pts[11] - pts[0])
    # Distance between Jan (idx 0) and Feb (idx 1)
    dist_jan_feb = np.linalg.norm(pts[0] - pts[1])
    assert abs(dist_dec_jan - dist_jan_feb) < 1e-6


def test_dtypes_and_zero_nans():
    grid_df = _make_3cell_grid()
    dates = pd.date_range("2015-01-03", "2015-01-15", freq="D")
    base_df = build_base_features(grid_df, dates)
    fire_df = build_fire_features(
        detections_df=pd.DataFrame(columns=["cell_id", "acq_date"]),
        grid_df=grid_df,
        start_date="2015-01-01",
        end_date="2015-01-15",
        eval_start="2015-01-03",
        eval_end="2015-01-15",
        n_rows=3,
        n_cols=4,
    )
    for col in ["fire_count_7d", "fire_count_30d", "neighbour_fire_7d"]:
        assert fire_df[col].dtype == np.float32
        assert fire_df[col].isna().sum() == 0

    for col in ["month_sin", "month_cos", "lat_c", "lon_c"]:
        assert base_df[col].dtype == np.float32
        assert base_df[col].isna().sum() == 0


def test_registry_completeness_for_day1():
    expected = [
        "fire_count_7d",
        "fire_count_30d",
        "neighbour_fire_7d",
        "month_sin",
        "month_cos",
        "lat_c",
        "lon_c",
    ]
    names = registry_names()
    for name in expected:
        assert name in names
        spec = get(name)
        assert spec.uses_future is False

    for spec in all_specs():
        assert spec.uses_future is False


@pytest.mark.skipif(not FIRE_TABLE_PATH.exists(), reason="Fire table not built yet")
def test_saved_fire_table_unchanged():
    df = pd.read_parquet(FIRE_TABLE_PATH)
    assert len(df) == 4684338
    assert int(df["n_det"].sum()) == 811133
    assert int(df["burning_prev3"].sum()) == 514620
    assert int(df["fire_next_48h"].sum()) == 399193
    assert int(df["new_fire_next_48h"].sum()) == 183647


@pytest.mark.skipif(not Path("data/processed/features_fire_base.parquet").exists(), reason="features_fire_base not built yet")
def test_saved_features_fire_base():
    df = pd.read_parquet("data/processed/features_fire_base.parquet")
    assert len(df) == 4684338
    for col in ["fire_count_7d", "fire_count_30d", "neighbour_fire_7d", "month_sin", "month_cos", "lat_c", "lon_c"]:
        assert col in df.columns
        assert df[col].dtype == np.float32
        assert df[col].isna().sum() == 0
