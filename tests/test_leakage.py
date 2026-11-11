from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.config import FIRE_TABLE_PATH, LABEL_HORIZON_DAYS
from src.features.fire_features import build_fire_features
from src.utils.fit_registry import (
    FitRecord,
    assert_fit_within_train,
    clear_registry,
    get_registered_fits,
    register_fit,
)


def test_fit_registry_boundary_checks():
    clear_registry()
    rec_valid = register_fit("valid_scaler", "2015-01-01", "2022-12-29")
    assert_fit_within_train(rec_valid)

    rec_invalid = FitRecord(name="leaky_model", first_date=date(2015, 1, 1), last_date=date(2023, 1, 1))
    with pytest.raises(AssertionError, match="Leakage detected"):
        assert_fit_within_train(rec_invalid)


def test_registry_all_fits_within_train():
    fits = get_registered_fits()
    for rec in fits:
        assert_fit_within_train(rec)


def test_shift_property_on_synthetic_sequence():
    # Helper: ensure quantity on day t depends only on detections <= t
    # For burning_prev3: sum over t-2, t-1, t
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    rng = np.random.default_rng(42)
    detections = []
    for d in dates:
        if rng.random() > 0.5:
            detections.append(d.date())

    def calc_burning_prev3(target_d: date, det_list: list[date]) -> bool:
        # Only considers dates in [target_d - 2d, target_d]
        valid_range = {target_d - timedelta(days=2), target_d - timedelta(days=1), target_d}
        return any(d in valid_range for d in det_list)

    target_date = dates[5].date()
    # Baseline with all detections
    val_full = calc_burning_prev3(target_date, detections)
    # Deleting all detections strictly after target_date
    det_truncated = [d for d in detections if d <= target_date]
    val_trunc = calc_burning_prev3(target_date, det_truncated)
    assert val_full == val_trunc

    # For label next2: depends on t+1 and t+2
    def calc_label_next2(target_d: date, det_list: list[date]) -> int:
        valid_future = {target_d + timedelta(days=1), target_d + timedelta(days=2)}
        return 1 if any(d in valid_future for d in det_list) else 0

    # Ensure deleting detections strictly after t+2 does not alter label
    det_trunc_after_t2 = [d for d in detections if d <= target_date + timedelta(days=2)]
    val_label_full = calc_label_next2(target_date, detections)
    val_label_trunc = calc_label_next2(target_date, det_trunc_after_t2)
    assert val_label_full == val_label_trunc


@pytest.mark.skipif(not FIRE_TABLE_PATH.exists(), reason="Fire table not built yet")
def test_shift_test_on_panel_sample():
    df = pd.read_parquet(FIRE_TABLE_PATH)
    clean_df = pd.read_parquet("data/processed/detections_clean.parquet")
    det_map = set(zip(clean_df["cell_id"], pd.to_datetime(clean_df["acq_date"]).dt.date))

    # Pick 200 random rows from fire table
    sample = df.sample(n=200, random_state=42)

    for _, row in sample.iterrows():
        c = row["cell_id"]
        t = row["date"].date()
        expected_burning = row["burning_prev3"]

        # Recompute burning_prev3 by inspecting detections strictly on t-2, t-1, t
        window_days = [t - timedelta(days=2), t - timedelta(days=1), t]
        actual_burning = any((c, d) in det_map for d in window_days)
        assert expected_burning == actual_burning


@pytest.mark.skipif(not FIRE_TABLE_PATH.exists(), reason="Fire table not built yet")
def test_window_leakage_and_splits():
    df = pd.read_parquet(FIRE_TABLE_PATH)

    train_dates = df[df["split"] == "train"]["date"]
    val_dates = df[df["split"] == "val"]["date"]
    test_dates = df[df["split"] == "test"]["date"]

    train_last = train_dates.max()
    val_first = val_dates.min()
    val_last = val_dates.max()
    test_first = test_dates.min()

    # Last date + LABEL_HORIZON_DAYS < first date of next split
    assert train_last + timedelta(days=LABEL_HORIZON_DAYS) < val_first
    assert val_last + timedelta(days=LABEL_HORIZON_DAYS) < test_first

    # Exactly 2 gap days strictly between splits
    assert (val_first - train_last).days - 1 == 2
    assert (test_first - val_last).days - 1 == 2


@pytest.mark.skipif(not FIRE_TABLE_PATH.exists(), reason="Fire table not built yet")
def test_panel_uniqueness():
    df = pd.read_parquet(FIRE_TABLE_PATH)

    # (cell_id, date) is unique
    assert not df.duplicated(subset=["cell_id", "date"]).any()

    # Every row has exactly one split
    assert df["split"].notnull().all()

    # No date has two different split values
    date_splits = df.groupby("date")["split"].nunique()
    assert (date_splits == 1).all()


def assert_prefix_invariant(
    full_df: pd.DataFrame,
    trunc_fn,
    feature_cols: list[str],
    cutoff_dates: list[date],
    date_col: str = "date",
) -> None:
    """Reusable helper verifying that truncating inputs at cutoff T does not change features <= T."""
    for cutoff in cutoff_dates:
        trunc_df = trunc_fn(cutoff)
        cutoff_ts = pd.Timestamp(cutoff)

        full_sub = full_df[full_df[date_col] <= cutoff_ts].sort_values(["cell_id", date_col]).reset_index(drop=True)
        trunc_sub = trunc_df[trunc_df[date_col] <= cutoff_ts].sort_values(["cell_id", date_col]).reset_index(drop=True)

        assert len(full_sub) == len(trunc_sub), f"Row count mismatch at cutoff {cutoff}"
        for col in feature_cols:
            np.testing.assert_allclose(
                full_sub[col].values,
                trunc_sub[col].values,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Prefix invariance violated for {col} at cutoff {cutoff}",
            )


def test_fire_features_prefix_invariance():
    # Synthetic 4x4 grid, 60 days, random detections
    rng = np.random.default_rng(42)
    grid_df = pd.DataFrame(
        {
            "cell_id": [f"r{r:02d}c{c:02d}" for r in range(4) for c in range(4)],
            "row": [r for r in range(4) for _ in range(4)],
            "col": [c for _ in range(4) for c in range(4)],
        }
    )
    all_dates = pd.date_range("2015-01-01", periods=60, freq="D")
    det_rows = []
    for _ in range(120):
        cell = grid_df["cell_id"].iloc[int(rng.integers(0, len(grid_df)))]
        dt = all_dates[int(rng.integers(0, len(all_dates)))].date()
        det_rows.append({"cell_id": cell, "acq_date": dt})
    detections_df = pd.DataFrame(det_rows)

    full_df = build_fire_features(
        detections_df=detections_df,
        grid_df=grid_df,
        start_date="2015-01-01",
        end_date="2015-03-01",
        eval_start="2015-01-03",
        eval_end="2015-03-01",
        n_rows=4,
        n_cols=4,
    )

    def make_trunc_fn(cutoff: date):
        sub_dets = detections_df[detections_df["acq_date"] <= cutoff].copy()
        return build_fire_features(
            detections_df=sub_dets,
            grid_df=grid_df,
            start_date="2015-01-01",
            end_date="2015-03-01",
            eval_start="2015-01-03",
            eval_end="2015-03-01",
            n_rows=4,
            n_cols=4,
        )

    # 5 random cutoff dates
    sample_idx = sorted(rng.choice(np.arange(5, 55), size=5, replace=False))
    cutoffs = [all_dates[i].date() for i in sample_idx]

    assert_prefix_invariant(
        full_df=full_df,
        trunc_fn=make_trunc_fn,
        feature_cols=["fire_count_7d", "fire_count_30d", "neighbour_fire_7d"],
        cutoff_dates=cutoffs,
    )


def test_full_builder_prefix_invariance():
    from src.features.build_features import ALL_FEATURE_NAMES, build_synthetic_subset_features

    rng = np.random.default_rng(42)
    grid_df = pd.DataFrame(
        {
            "cell_id": [f"r{r:02d}c{c:02d}" for r in range(4) for c in range(4)],
            "row": [r for r in range(4) for _ in range(4)],
            "col": [c for _ in range(4) for c in range(4)],
            "lat_c": [20.0 + 0.5 * r for r in range(4) for _ in range(4)],
            "lon_c": [78.0 + 0.5 * c for _ in range(4) for c in range(4)],
        }
    )

    weather_dates = pd.date_range("2014-12-01", "2015-03-01", freq="D")
    n_days = len(weather_dates)
    n_cells = len(grid_df)
    tmax_c = rng.uniform(18.0, 38.0, size=(n_days, n_cells)).astype(np.float32)
    precip_mm = rng.exponential(scale=2.0, size=(n_days, n_cells)).astype(np.float32)
    mean_annual_mm = np.full(n_cells, 1100.0, dtype=np.float32)

    all_study_dates = pd.date_range("2015-01-01", "2015-03-01", freq="D")
    det_rows = []
    for _ in range(100):
        cell = grid_df["cell_id"].iloc[int(rng.integers(0, n_cells))]
        dt = all_study_dates[int(rng.integers(0, len(all_study_dates)))].date()
        det_rows.append({"cell_id": cell, "acq_date": dt})
    detections_df = pd.DataFrame(det_rows)

    eval_dates = pd.date_range("2015-01-03", "2015-03-01", freq="D")
    ft_rows = []
    for cell in grid_df["cell_id"]:
        for dt in eval_dates:
            ft_rows.append(
                {
                    "cell_id": cell,
                    "date": dt,
                    "n_det": 0,
                    "frp_sum": np.float32(0.0),
                    "burning_prev3": False,
                    "new_fire_next_48h": 0,
                    "fire_next_48h": 0,
                    "split": "train",
                }
            )
    fire_table_df = pd.DataFrame(ft_rows)

    full_df = build_synthetic_subset_features(
        detections_df=detections_df,
        grid_df=grid_df,
        tmax_c=tmax_c,
        precip_mm=precip_mm,
        weather_dates=weather_dates,
        mean_annual_mm=mean_annual_mm,
        fire_table_df=fire_table_df,
        start_date="2015-01-01",
        end_date="2015-03-01",
        eval_start="2015-01-03",
        eval_end="2015-03-01",
        n_rows=4,
        n_cols=4,
    )

    def make_trunc_fn(cutoff: date):
        sub_dets = detections_df[detections_df["acq_date"] <= cutoff].copy()
        mask_future = weather_dates.date > cutoff
        tmax_trunc = tmax_c.copy()
        precip_trunc = precip_mm.copy()
        tmax_trunc[mask_future] = 55.0
        precip_trunc[mask_future] = 250.0
        ft_trunc = fire_table_df.copy()
        ft_trunc.loc[ft_trunc["date"].dt.date > cutoff, "new_fire_next_48h"] = 1
        return build_synthetic_subset_features(
            detections_df=sub_dets,
            grid_df=grid_df,
            tmax_c=tmax_trunc,
            precip_mm=precip_trunc,
            weather_dates=weather_dates,
            mean_annual_mm=mean_annual_mm,
            fire_table_df=ft_trunc,
            start_date="2015-01-01",
            end_date="2015-03-01",
            eval_start="2015-01-03",
            eval_end="2015-03-01",
            n_rows=4,
            n_cols=4,
        )

    sample_idx = sorted(rng.choice(np.arange(5, len(eval_dates) - 5), size=3, replace=False))
    cutoffs = [eval_dates[i].date() for i in sample_idx]

    assert_prefix_invariant(
        full_df=full_df,
        trunc_fn=make_trunc_fn,
        feature_cols=ALL_FEATURE_NAMES,
        cutoff_dates=cutoffs,
    )


def test_fit_registry_day4_constraints():
    clear_registry()
    rec_year_end = register_fit(
        "mean_annual_rainfall",
        "2015-01-01",
        "2022-12-31",
        metadata={"allow_year_end": True},
    )
    assert_fit_within_train(rec_year_end)

    rec_leaky = FitRecord(name="bad_scaler", first_date=date(2015, 1, 1), last_date=date(2023, 1, 1))
    with pytest.raises(AssertionError, match="Leakage detected"):
        assert_fit_within_train(rec_leaky)


