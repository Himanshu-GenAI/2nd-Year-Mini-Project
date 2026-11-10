from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.kbdi import compute_kbdi
from src.features.registry import all_specs, get, registry_names
from src.features.weather_features import (
    WEATHER_FEATURE_NAMES,
    compute_days_since_rain,
    compute_weather_and_drought_features,
    mean_annual_rainfall_mm,
)
from src.utils.fit_registry import clear_registry, get_registered_fits
from tests.test_leakage import assert_prefix_invariant


def test_hand_computed_lags_and_sums():
    # Create 40 days starting 2014-12-25 to 2015-02-02 (so 2015-01-03 is index 9 >= 7)
    dates = pd.date_range("2014-12-25", periods=40, freq="D")
    n_days = len(dates)
    # Ramp tmax: 10.0, 11.0, 12.0, ...
    tmax = (10.0 + np.arange(n_days, dtype=np.float32)).reshape(-1, 1)
    # Constant precip 2.0 mm/day
    precip = np.full((n_days, 1), 2.0, dtype=np.float32)
    mean_rain = np.array([1000.0], dtype=np.float32)

    df = compute_weather_and_drought_features(
        tmax_c=tmax,
        precip_mm=precip,
        mean_annual_mm=mean_rain,
        dates=dates,
        cell_ids=["c01"],
        eval_start="2015-01-03",
        eval_end="2015-02-02",
    )

    # Check index 9 (2015-01-03) -> first row of df
    row0 = df.iloc[0]
    assert row0["date"] == pd.Timestamp("2015-01-03")
    # At idx=9: tmax_t0 = 19.0, tmax_t1 = 18.0, tmax_t3 = 16.0, tmax_t7 = 12.0
    assert row0["tmax_t0"] == 19.0
    assert row0["tmax_t1"] == 18.0
    assert row0["tmax_t3"] == 16.0
    assert row0["tmax_t7"] == 12.0
    # 7-day mean tmax over idx 3..9: (13+14+15+16+17+18+19)/7 = 16.0
    assert abs(row0["tmax_mean_7d"] - 16.0) < 1e-5
    # 7-day precip sum: 7 * 2.0 = 14.0
    assert row0["precip_sum_7d"] == 14.0

    # Check index 35 (36th day, index 26 in eval): full 30-day precip sum = 30 * 2.0 = 60.0
    row_late = df.iloc[25]
    assert row_late["precip_sum_30d"] == 60.0


def test_days_since_rain_thresholds_and_caps():
    # 400 days dry except specific rain days
    precip = np.zeros((400, 1), dtype=np.float32)
    # Day 0: 0 mm -> dsr = 1 (days since day 0 + 1)
    # Day 2: 2.5 mm (NOT > 2.5, so NOT counted!) -> dsr = 3
    # Day 3: 2.6 mm (> 2.5, counted!) -> dsr = 0
    # Days 4..399: 0 mm -> dsr rises 1, 2, ..., capped at 365
    precip[2, 0] = 2.5
    precip[3, 0] = 2.6

    dsr = compute_days_since_rain(precip, threshold_mm=2.5, cap=365)
    assert dsr[0, 0] == 1.0
    assert dsr[1, 0] == 2.0
    assert dsr[2, 0] == 3.0  # 2.5 mm ignored
    assert dsr[3, 0] == 0.0  # 2.6 mm counted
    assert dsr[4, 0] == 1.0
    assert dsr[368, 0] == 365.0
    assert dsr[399, 0] == 365.0  # Capped at 365


def test_kbdi_change_7d_exact():
    dates = pd.date_range("2014-12-20", periods=30, freq="D")
    tmax = np.full((30, 1), 35.0, dtype=np.float32)
    precip = np.zeros((30, 1), dtype=np.float32)
    mean_rain = np.array([1000.0], dtype=np.float32)

    kbdi_full = compute_kbdi(tmax, precip, mean_rain)
    df = compute_weather_and_drought_features(
        tmax_c=tmax,
        precip_mm=precip,
        mean_annual_mm=mean_rain,
        dates=dates,
        cell_ids=["c01"],
        eval_start="2015-01-03",
        eval_end="2015-01-18",
    )

    # 2015-01-03 is index 14 in dates
    expected_change = float(kbdi_full[14, 0] - kbdi_full[7, 0])
    assert abs(df.iloc[0]["kbdi_change_7d"] - expected_change) < 1e-4


def test_mean_annual_rainfall_fit_record(tmp_path: Path):
    clear_registry()
    dates = pd.date_range("2014-01-01", "2025-12-31", freq="D")
    precip = np.zeros((len(dates), 1), dtype=np.float32)
    # Set 1.0 mm every day in 2015-2022 (8 years = 2922 days -> 2922 / 8 = 365.25 mm/yr)
    mask_15_22 = (dates.year >= 2015) & (dates.year <= 2022)
    precip[mask_15_22, 0] = 1.0
    # Put huge rain in 2023-2025 to prove it's excluded
    precip[dates.year >= 2023, 0] = 500.0

    save_p = tmp_path / "rain.parquet"
    res = mean_annual_rainfall_mm(precip, dates, cell_ids=["c01"], save_path=save_p)
    assert abs(res[0] - (mask_15_22.sum() / 8.0)) < 1e-4

    fits = [r for r in get_registered_fits() if r.name == "mean_annual_rainfall"]
    assert len(fits) > 0
    assert fits[-1].last_date == date(2022, 12, 31)
    assert save_p.exists()


def test_weather_features_prefix_invariance():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2014-12-01", periods=90, freq="D")
    n_days = len(dates)
    tmax = rng.uniform(18.0, 38.0, size=(n_days, 2)).astype(np.float32)
    precip = rng.exponential(scale=2.0, size=(n_days, 2)).astype(np.float32)
    # Fixed mean annual rainfall (fitted on 2015-2022 only in production)
    mean_rain = np.array([1000.0, 1500.0], dtype=np.float32)

    full_df = compute_weather_and_drought_features(
        tmax_c=tmax,
        precip_mm=precip,
        mean_annual_mm=mean_rain,
        dates=dates,
        cell_ids=["c01", "c02"],
        eval_start="2015-01-03",
        eval_end="2015-02-28",
    )

    def trunc_fn(cutoff: date):
        # Replace future values after cutoff with distinct extreme values to prove causality
        mask_future = dates.date > cutoff
        tmax_trunc = tmax.copy()
        precip_trunc = precip.copy()
        tmax_trunc[mask_future] = 55.0
        precip_trunc[mask_future] = 250.0

        return compute_weather_and_drought_features(
            tmax_c=tmax_trunc,
            precip_mm=precip_trunc,
            mean_annual_mm=mean_rain,
            dates=dates,
            cell_ids=["c01", "c02"],
            eval_start="2015-01-03",
            eval_end="2015-02-28",
        )

    eval_dates = pd.date_range("2015-01-03", "2015-02-28", freq="D")
    sample_idx = sorted(rng.choice(np.arange(len(eval_dates)), size=5, replace=False))
    cutoffs = [eval_dates[i].date() for i in sample_idx]

    assert_prefix_invariant(
        full_df=full_df,
        trunc_fn=trunc_fn,
        feature_cols=WEATHER_FEATURE_NAMES,
        cutoff_dates=cutoffs,
    )


def test_weather_registry_and_dtypes():
    names = registry_names()
    for col in WEATHER_FEATURE_NAMES:
        assert col in names
        spec = get(col)
        assert spec.uses_future is False
        assert spec.dtype == "float32"

    for spec in all_specs():
        assert spec.uses_future is False

