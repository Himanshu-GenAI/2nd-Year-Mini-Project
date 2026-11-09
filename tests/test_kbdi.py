import time

import numpy as np
import pandas as pd

from src.features.kbdi import (
    compute_kbdi,
    compute_kbdi_for_study_period,
)


def test_hot_dry_stretch_diminishing_increments():
    # 35 C = 95 F. Precip = 0. Mean annual rain = 1000 mm (~39.37 in).
    n_days = 7
    tmax = np.full((n_days, 1), 35.0)
    precip = np.zeros((n_days, 1))
    mean_annual = np.array([1000.0])

    q_series = compute_kbdi(tmax, precip, mean_annual, q0=100.0)

    # Q must strictly rise
    diffs = np.diff(q_series[:, 0])
    assert (diffs > 0).all()

    # Compare increments starting from high Q (e.g. Q=750 vs Q=100)
    q_high = compute_kbdi(tmax, precip, mean_annual, q0=750.0)
    diff_high = q_high[1, 0] - q_high[0, 0]
    diff_low = q_series[1, 0] - q_series[0, 0]
    # Increases must be smaller as Q approaches 800
    assert diff_high < diff_low


def test_heavy_rain_day_net_interception():
    # Single heavy rain day of 40 mm after dry state
    # 40 mm = 40 / 25.4 = 1.574803 in
    # Net rain = 1.574803 - 0.20 = 1.374803 in
    # Q drop = 100 * 1.374803 = 137.4803 units
    tmax = np.array([[10.0]])  # 10 C = 50 F -> dQ = 0
    precip = np.array([[40.0]])
    mean_annual = np.array([1000.0])

    q0 = 500.0
    q_out = compute_kbdi(tmax, precip, mean_annual, q0=q0)
    expected_q = max(q0 - 100.0 * (40.0 / 25.4 - 0.20), 0.0)

    assert abs(q_out[0, 0] - expected_q) < 1e-4


def test_two_consecutive_rain_days_single_interception():
    # Day 1: 10 mm (0.3937 in). Net = 0.3937 - 0.20 = 0.1937 in.
    # Day 2: 10 mm (0.3937 in). Spell total = 20 mm (0.7874 in).
    # Net on Day 2: (0.7874 - 0.20) - (0.3937 - 0.20) = 0.3937 in (full amount, no second 0.20 subtraction)
    tmax = np.full((2, 1), 10.0)  # 50 F -> dQ = 0
    precip = np.full((2, 1), 10.0)
    mean_annual = np.array([1000.0])

    q0 = 400.0
    q_out = compute_kbdi(tmax, precip, mean_annual, q0=q0)

    total_net_rain_in = (20.0 / 25.4) - 0.20
    expected_final_q = q0 - 100.0 * total_net_rain_in
    assert abs(q_out[1, 0] - expected_final_q) < 1e-4


def test_sub_threshold_rain_ignored():
    # Rain below 0.254 mm (0.01 in) does not count as rain day
    tmax = np.array([[10.0]])
    precip = np.array([[0.20]])  # < 0.254 mm
    mean_annual = np.array([1000.0])

    q_out = compute_kbdi(tmax, precip, mean_annual, q0=300.0)
    assert q_out[0, 0] == 300.0


def test_temperature_floor_no_increment():
    # Exactly 50 F (10 C) or below gives dQ = 0
    tmax = np.array([[10.0], [5.0], [-5.0]])
    precip = np.zeros((3, 1))
    mean_annual = np.array([1000.0])

    q_out = compute_kbdi(tmax, precip, mean_annual, q0=250.0)
    assert (q_out == 250.0).all()


def test_random_inputs_stay_bounded():
    # 10,000 steps property test with fixed seed
    rng = np.random.default_rng(42)
    n = 10000
    tmax = rng.uniform(-10.0, 50.0, size=(n, 1))
    precip = rng.exponential(scale=5.0, size=(n, 1))
    mean_annual = np.array([1200.0])

    q_out = compute_kbdi(tmax, precip, mean_annual, q0=0.0)
    assert (q_out >= 0.0).all()
    assert (q_out <= 800.0).all()


def test_cell_independence_and_repeatability():
    tmax = np.array([[30.0, 30.0], [25.0, 35.0]])
    precip = np.array([[0.0, 15.0], [5.0, 0.0]])
    mean_annual = np.array([800.0, 1200.0])

    # Run in batch
    res_batch = compute_kbdi(tmax, precip, mean_annual, q0=100.0)
    # Run cell 0 alone
    res_single = compute_kbdi(tmax[:, :1], precip[:, :1], mean_annual[:1], q0=100.0)

    assert np.allclose(res_batch[:, 0], res_single[:, 0])


def test_rainfall_climate_and_temperature_effects_on_dq():
    # In Keetch-Byram (1968), denominator is (1 + 10.88 * exp(-0.0441 * R_in)).
    # As R_in increases, exp(-0.0441 * R_in) decreases -> denominator decreases -> dQ increases.
    # Higher annual precipitation climates have higher potential evapotranspiration drying rate.
    tmax = np.array([[38.0, 38.0]])
    precip = np.zeros((1, 2))
    mean_annual = np.array([500.0, 2000.0])

    q_out = compute_kbdi(tmax, precip, mean_annual, q0=200.0)
    # Cell 1 (humid climate R=2000mm) has lower denominator than Cell 0 (arid R=500mm), so dQ is higher
    assert q_out[0, 1] > q_out[0, 0]

    # Two cells: same R, cell 0 (T=40C), cell 1 (T=25C)
    tmax2 = np.array([[40.0, 25.0]])
    mean_annual2 = np.array([1000.0, 1000.0])
    q_out2 = compute_kbdi(tmax2, precip, mean_annual2, q0=200.0)
    assert q_out2[0, 0] > q_out2[0, 1]


def test_study_period_wrapper_discards_2014():
    dates = pd.date_range("2014-01-01", "2015-01-05", freq="D")
    n_days = len(dates)
    tmax = np.full((n_days, 1), 25.0)
    precip = np.zeros((n_days, 1))
    mean_annual = np.array([1000.0])

    trimmed_q, trimmed_dates = compute_kbdi_for_study_period(tmax, precip, mean_annual, dates, q0=0.0)
    assert trimmed_dates.min() == pd.to_datetime("2015-01-01")
    assert len(trimmed_q) == 5  # Jan 1 to Jan 5 2015


def test_kbdi_speed_bench():
    # 4,383 days x 1,167 cells
    n_days = 4383
    n_cells = 1167
    rng = np.random.default_rng(42)
    tmax = rng.uniform(15.0, 42.0, size=(n_days, n_cells)).astype(np.float32)
    precip = rng.exponential(scale=3.0, size=(n_days, n_cells)).astype(np.float32)
    mean_annual = rng.uniform(600.0, 2500.0, size=n_cells).astype(np.float32)

    start_t = time.perf_counter()
    q_full = compute_kbdi(tmax, precip, mean_annual, q0=0.0)
    duration = time.perf_counter() - start_t

    print(f"KBDI full matrix ({n_days} x {n_cells}) computed in {duration:.2f}s")
    assert duration < 120.0  # Must be under 2 minutes
    assert q_full.shape == (n_days, n_cells)
