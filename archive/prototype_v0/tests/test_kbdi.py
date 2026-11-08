"""Unit tests for Keetch-Byram Drought Index (KBDI) calculations."""

import pandas as pd

from src.data.kbdi import (
    classify_kbdi_danger,
    compute_kbdi_timeseries,
    update_kbdi_daily,
)


def test_kbdi_increases_during_hot_dry_weather():
    initial = 200.0
    # 38°C hot day with 0 rain should increase drought index
    next_day = update_kbdi_daily(
        kbdi_prev=initial,
        temp_max_c=38.0,
        rain_mm=0.0,
        annual_precip_mm=1000.0,
    )
    assert next_day > initial


def test_kbdi_rain_interception():
    initial = 400.0
    # Rainfall below canopy interception threshold (e.g. 3 mm < 5.08 mm) should not reduce KBDI
    after_light_rain = update_kbdi_daily(
        kbdi_prev=initial,
        temp_max_c=20.0,
        rain_mm=3.0,
        rain_interception_mm=5.08,
    )
    # KBDI should either stay constant or slightly increase due to modest temp
    assert after_light_rain >= initial


def test_kbdi_decreases_with_heavy_rain():
    initial = 500.0
    # 50 mm heavy downpour should significantly reduce drought index
    after_storm = update_kbdi_daily(
        kbdi_prev=initial,
        temp_max_c=22.0,
        rain_mm=50.0,
        rain_interception_mm=5.08,
    )
    assert after_storm < initial


def test_kbdi_bounds_clamped():
    # Never below 0
    val_low = update_kbdi_daily(kbdi_prev=10.0, temp_max_c=10.0, rain_mm=200.0)
    assert val_low >= 0.0

    # Never above 800
    val_high = update_kbdi_daily(kbdi_prev=799.0, temp_max_c=48.0, rain_mm=0.0)
    assert val_high <= 800.0


def test_kbdi_timeseries():
    df = pd.DataFrame(
        {
            "temp_max_c": [30.0, 32.0, 35.0, 25.0],
            "precip_24h_mm": [0.0, 0.0, 0.0, 40.0],
        }
    )
    kbdi_series = compute_kbdi_timeseries(df, initial_kbdi=300.0)
    assert len(kbdi_series) == 4
    # First 3 days increase, 4th day drops
    assert kbdi_series.iloc[2] > kbdi_series.iloc[0]
    assert kbdi_series.iloc[3] < kbdi_series.iloc[2]


def test_classify_kbdi_danger():
    assert classify_kbdi_danger(150.0) == "Low"
    assert classify_kbdi_danger(350.0) == "Moderate"
    assert classify_kbdi_danger(550.0) == "High"
    assert classify_kbdi_danger(750.0) == "Extreme"
