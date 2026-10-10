"""Critical Data Leakage and Lookahead Safeguard Verification Suite.

Guarantees that:
1. Target labels at day t strictly represent future observation windows (t+1, t+3).
2. Historical lag features at day t only reference prior observations (<= t-1).
3. Temporal train/val/test splits maintain strict chronological ordering with no future leakage.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_fire_lag_features,
)
from src.features.labels import generate_fire_labels
from src.models.train import split_temporal_dataset


@pytest.fixture
def sample_temporal_panel():
    """Create deterministic 10-day panel across two grid cells."""
    dates = pd.date_range("2021-01-01", "2021-01-10", freq="D")
    records = []
    for gid in ["cell_A", "cell_B"]:
        for i, d in enumerate(dates):
            # Fire occurs only on day 5 and day 8
            f_count = 1 if i in [4, 7] else 0
            records.append(
                {
                    "grid_id": gid,
                    "date": d,
                    "fire_count": f_count,
                    "temp_max_c": 30.0 + i,
                    "precip_24h_mm": 0.0,
                    "relative_humidity_pct": 40.0,
                    "wind_speed_kmh": 15.0,
                    "kbdi": 300.0 + i * 10,
                }
            )
    return pd.DataFrame(records)


def test_target_label_strictly_forward_shifted(sample_temporal_panel):
    """Verify target at day t is equal to fire_count at day t+1, NOT day t."""
    labeled = generate_fire_labels(
        sample_temporal_panel,
        horizon_days=1,
        fire_count_col="fire_count",
        date_col="date",
        grid_col="grid_id",
        drop_unlabeled_future=True,
    )

    cell_a = labeled[labeled["grid_id"] == "cell_A"].sort_values("date").reset_index(drop=True)

    # Day 4 (2021-01-04): fire_count today is 0, but fire occurs tomorrow (day 5)
    row_day4 = cell_a[cell_a["date"] == "2021-01-04"].iloc[0]
    assert row_day4["fire_count"] == 0
    assert row_day4["fire_occurred_next_day"] == 1  # Target should correctly anticipate tomorrow

    # Day 5 (2021-01-05): fire occurs today (1), but tomorrow (day 6) fire is 0
    row_day5 = cell_a[cell_a["date"] == "2021-01-05"].iloc[0]
    assert row_day5["fire_count"] == 1
    assert row_day5["fire_occurred_next_day"] == 0  # Tomorrow has no fire


def test_last_day_dropped_to_prevent_unlabeled_leakage(sample_temporal_panel):
    """The final observed day has no ground-truth t+1; it must be excluded from training."""
    labeled = generate_fire_labels(
        sample_temporal_panel,
        horizon_days=1,
        drop_unlabeled_future=True,
    )
    max_original_date = sample_temporal_panel["date"].max()
    max_labeled_date = labeled["date"].max()

    assert max_labeled_date < max_original_date
    assert (max_original_date - max_labeled_date).days == 1


def test_lag_features_strictly_backward_looking(sample_temporal_panel):
    """Verify lag features on day t do NOT incorporate events occurring on day t."""
    lagged = add_fire_lag_features(
        sample_temporal_panel,
        grid_col="grid_id",
        date_col="date",
        fire_count_col="fire_count",
        lag_days=[1, 3],
    )

    cell_a = lagged[lagged["grid_id"] == "cell_A"].sort_values("date").reset_index(drop=True)

    # Day 5: Fire occurs today!
    row_day5 = cell_a[cell_a["date"] == "2021-01-05"].iloc[0]
    # Yesterday's lag must STILL be 0 because fire only ignited today
    assert row_day5["fire_count_lag_1d"] == 0

    # Day 6: Now yesterday's lag should reflect day 5's fire
    row_day6 = cell_a[cell_a["date"] == "2021-01-06"].iloc[0]
    assert row_day6["fire_count_lag_1d"] == 1


def test_strict_temporal_split_ordering():
    """Verify chronological non-overlap: max(train) < min(val) < max(val) < min(test)."""
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "val": np.arange(len(dates))})

    train_df, val_df, test_df = split_temporal_dataset(
        df,
        date_col="date",
        train_end="2020-12-31",
        val_start="2021-01-01",
        val_end="2021-12-31",
        test_start="2022-01-01",
    )

    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()
    assert len(set(train_df["date"]).intersection(set(val_df["date"]))) == 0
    assert len(set(val_df["date"]).intersection(set(test_df["date"]))) == 0
