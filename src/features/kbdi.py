"""Keetch-Byram Drought Index (KBDI) Recursive Calculation Engine.

Implements the official USDA Forest Service (Keetch & Byram, 1968, Research Paper SE-38)
equations modeling cumulative duff/upper-soil moisture deficit (0 to 800) with
rain interception thresholding and vectorized multi-cell computation.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    KBDI_Q0,
    MEAN_RAIN_YEARS,
    RAIN_DAY_MM,
    RAIN_SPELL_FIRST_IN,
    STUDY_START,
)
from src.utils.fit_registry import register_fit

MM_PER_INCH = 25.4


def celsius_to_fahrenheit(celsius: float | np.ndarray) -> float | np.ndarray:
    """Convert temperature from Celsius to Fahrenheit."""
    return celsius * 1.8 + 32.0


def mm_to_inches(mm: float | np.ndarray) -> float | np.ndarray:
    """Convert rainfall from millimeters to inches."""
    return mm / MM_PER_INCH


def mean_annual_rainfall_mm(
    weather_df: pd.DataFrame,
    cell_col: str = "cell_id",
    date_col: str = "date",
    precip_col: str = "precipitation_sum",
) -> pd.Series:
    """Calculate mean annual precipitation per cell across training years (2015-2022).

    Registers the fit period in the fit registry to prevent leakage.
    """
    df = weather_df.copy()
    dates = pd.to_datetime(df[date_col])
    years = dates.dt.year

    start_year, end_year = MEAN_RAIN_YEARS
    mask = (years >= start_year) & (years <= end_year)
    train_weather = df[mask]

    first_date = dates[mask].min().date()
    last_date = dates[mask].max().date()
    num_years = float(end_year - start_year + 1)

    register_fit(
        name="mean_annual_rainfall",
        first_date=first_date,
        last_date=last_date,
        metadata={"years": MEAN_RAIN_YEARS, "num_years": num_years},
    )

    cell_totals = train_weather.groupby(cell_col, observed=True)[precip_col].sum()
    mean_annual = cell_totals / num_years
    return mean_annual


def compute_kbdi(
    tmax_c: np.ndarray,
    precip_mm: np.ndarray,
    mean_annual_mm: np.ndarray,
    q0: float = KBDI_Q0,
) -> np.ndarray:
    """Vectorized calculation of KBDI for multiple cells over time.

    Args:
        tmax_c: 2D array of maximum temperature (°C) of shape (n_days, n_cells).
        precip_mm: 2D array of daily precipitation (mm) of shape (n_days, n_cells).
        mean_annual_mm: 1D array of mean annual precipitation (mm) of shape (n_cells,).
        q0: Initial KBDI value on day 0 (default 0).

    Returns:
        2D array of KBDI values (0 to 800) of shape (n_days, n_cells).
    """
    n_days, n_cells = tmax_c.shape

    # Unit conversions
    tmax_f = celsius_to_fahrenheit(tmax_c)
    precip_in = mm_to_inches(precip_mm)
    r_in = mm_to_inches(mean_annual_mm)

    # Precompute denominator factor: (1 + 10.88 * exp(-0.0441 * R_in))
    # Shape: (n_cells,)
    denom_factor = 1.0 + 10.88 * np.exp(-0.0441 * r_in)

    kbdi_out = np.zeros((n_days, n_cells), dtype=np.float32)

    # Current state vectors across all cells
    q = np.full(n_cells, float(q0), dtype=np.float64)
    spell_total_in = np.zeros(n_cells, dtype=np.float64)

    rain_day_thresh_in = mm_to_inches(RAIN_DAY_MM)
    interception_in = RAIN_SPELL_FIRST_IN

    for day in range(n_days):
        p_day_in = precip_in[day]
        t_day_f = tmax_f[day]

        # 1. Rain processing
        is_rain_day = p_day_in >= rain_day_thresh_in

        # Update spell total
        prev_spell_in = np.where(is_rain_day, spell_total_in, 0.0)
        curr_spell_in = np.where(is_rain_day, prev_spell_in + p_day_in, 0.0)
        spell_total_in = curr_spell_in

        # Net rain: max(curr_spell - 0.20, 0) - max(prev_spell - 0.20, 0)
        net_rain_in = np.maximum(curr_spell_in - interception_in, 0.0) - np.maximum(prev_spell_in - interception_in, 0.0)

        # Apply rain reduction
        q = np.maximum(q - 100.0 * net_rain_in, 0.0)

        # 2. Drought factor increment dQ
        # Only where T_F > 50.0 °F
        temp_active = t_day_f > 50.0
        # Term: (0.968 * exp(0.0486 * T_F) - 8.30) * 0.001
        temp_term = np.where(
            temp_active,
            (0.968 * np.exp(0.0486 * t_day_f) - 8.30) * 0.001,
            0.0,
        )

        dq = np.where(
            temp_active,
            ((800.0 - q) * temp_term) / denom_factor,
            0.0,
        )
        # Avoid negative dQ if temperature is barely above 50 F and temp_term < 0
        dq = np.maximum(dq, 0.0)

        q = np.clip(q + dq, 0.0, 800.0)
        kbdi_out[day] = q

    return kbdi_out


def compute_kbdi_for_study_period(
    tmax_c: np.ndarray,
    precip_mm: np.ndarray,
    mean_annual_mm: np.ndarray,
    dates: pd.DatetimeIndex,
    q0: float = KBDI_Q0,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Calculate KBDI with 2014 spin-up and discard all 2014 spin-up dates."""
    kbdi_full = compute_kbdi(tmax_c, precip_mm, mean_annual_mm, q0=q0)

    # Discard dates before STUDY_START (2015-01-01)
    study_start_d = date.fromisoformat(STUDY_START)
    mask = [d.date() >= study_start_d for d in dates]

    trimmed_kbdi = kbdi_full[mask]
    trimmed_dates = dates[mask]
    return trimmed_kbdi, trimmed_dates
