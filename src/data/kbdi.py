"""Keetch-Byram Drought Index (KBDI) Implementation.

Mathematical formulation of the cumulative wildfire danger rating based on
daily maximum temperature, precipitation, and mean annual rainfall.
Scale ranges from 0 (saturated soil/no drought) to 800 (extreme drought).
"""

import numpy as np
import pandas as pd

from src.config import (
    ANNUAL_PRECIP_DEFAULT_MM,
    KBDI_DEFAULT_INIT,
    KBDI_MAX,
    KBDI_MIN,
    KBDI_THRESHOLDS,
    RAIN_INTERCEPTION_THRESHOLD_MM,
)


def compute_drought_factor(
    kbdi_prev: float,
    temp_max_c: float,
    annual_precip_mm: float = ANNUAL_PRECIP_DEFAULT_MM,
) -> float:
    """Calculate the daily KBDI incremental drought factor (dF).

    Args:
        kbdi_prev: KBDI on previous day (0-800).
        temp_max_c: Maximum temperature in Celsius.
        annual_precip_mm: Mean annual precipitation in mm.

    Returns:
        Drought factor increment (points).
    """
    temp_max_f = temp_max_c * 1.8 + 32.0

    # Evapotranspiration is negligible at or below 50°F (10°C)
    if temp_max_f <= 50.0:
        return 0.0

    annual_precip_in = annual_precip_mm / 25.4

    # Classic Keetch-Byram evapotranspiration formulation
    numerator = (800.0 - kbdi_prev) * (0.968 * np.exp(0.0486 * temp_max_f) - 8.30) * 1e-3
    denominator = 1.0 + 10.88 * np.exp(-0.001736 * annual_precip_in)

    df = numerator / denominator
    return float(max(0.0, df))


def update_kbdi_daily(
    kbdi_prev: float,
    temp_max_c: float,
    rain_mm: float,
    annual_precip_mm: float = ANNUAL_PRECIP_DEFAULT_MM,
    rain_interception_mm: float = RAIN_INTERCEPTION_THRESHOLD_MM,
) -> float:
    """Compute single-day update for KBDI.

    Args:
        kbdi_prev: Previous day's KBDI value [0, 800].
        temp_max_c: Maximum temperature in Celsius.
        rain_mm: Total 24-hr rainfall in mm.
        annual_precip_mm: Mean annual rainfall in mm.
        rain_interception_mm: Canopy/litter interception threshold in mm (default 5.08 mm = 0.20 in).

    Returns:
        Updated KBDI value clamped to [0, 800].
    """
    # Rain reduction: only precipitation exceeding interception threshold penetrates
    if rain_mm > rain_interception_mm:
        net_rain_in = (rain_mm - rain_interception_mm) / 25.4
        # 1 inch of rain reduces KBDI by 100 points
        rain_reduction = net_rain_in * 100.0
    else:
        rain_reduction = 0.0

    kbdi_after_rain = max(KBDI_MIN, kbdi_prev - rain_reduction)

    # Evaporative drying factor
    df = compute_drought_factor(kbdi_after_rain, temp_max_c, annual_precip_mm)

    new_kbdi = np.clip(kbdi_after_rain + df, KBDI_MIN, KBDI_MAX)
    return float(new_kbdi)


def compute_kbdi_timeseries(
    weather_df: pd.DataFrame,
    initial_kbdi: float = KBDI_DEFAULT_INIT,
    annual_precip_mm: float = ANNUAL_PRECIP_DEFAULT_MM,
    temp_col: str = "temp_max_c",
    rain_col: str = "precip_24h_mm",
) -> pd.Series:
    """Compute sequential KBDI values over an entire meteorological timeseries.

    Args:
        weather_df: DataFrame sorted chronologically containing temp and rain columns.
        initial_kbdi: Starting KBDI value.
        annual_precip_mm: Average annual precipitation.
        temp_col: Column name for maximum daily temperature in Celsius.
        rain_col: Column name for 24-hour rainfall in mm.

    Returns:
        Pandas Series of computed KBDI values.
    """
    temps = weather_df[temp_col].values
    rains = weather_df[rain_col].values
    n = len(weather_df)

    kbdi_values = np.zeros(n, dtype=np.float64)
    current_kbdi = initial_kbdi

    for i in range(n):
        current_kbdi = update_kbdi_daily(
            kbdi_prev=current_kbdi,
            temp_max_c=float(temps[i]),
            rain_mm=float(rains[i]),
            annual_precip_mm=annual_precip_mm,
        )
        kbdi_values[i] = round(current_kbdi, 2)

    return pd.Series(kbdi_values, index=weather_df.index, name="kbdi")


def classify_kbdi_danger(kbdi_value: float) -> str:
    """Map numeric KBDI value to standard forestry danger classification.

    Args:
        kbdi_value: KBDI score in [0, 800].

    Returns:
        One of 'Low', 'Moderate', 'High', 'Extreme'.
    """
    for category, (low, high) in KBDI_THRESHOLDS.items():
        if low <= kbdi_value <= high:
            return category
    return "Extreme" if kbdi_value > 800 else "Low"
