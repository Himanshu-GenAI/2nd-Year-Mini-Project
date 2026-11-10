"""Weather and KBDI drought feature engineering.

Loads cached cell Parquets without interpolation, calculates training-period
mean annual precipitation, computes causal lag/rolling weather metrics and
KBDI trajectories over 2014-2025, and exports aligned study-period features.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    KBDI_Q0,
    MEAN_RAIN_YEARS,
    PROCESSED_DIR,
    TEST_END,
    TRAIN_START,
    WEATHER_DIR,
    WEATHER_END,
    WEATHER_N_DAYS,
    WEATHER_START,
)
from src.data.grid import load_grid
from src.features.kbdi import compute_kbdi
from src.utils.fit_registry import register_fit

logger = logging.getLogger(__name__)

MEAN_ANNUAL_RAIN_PATH = PROCESSED_DIR / "mean_annual_rainfall.parquet"
FEATURES_WEATHER_PARTIAL_PATH = PROCESSED_DIR / "features_weather_partial.parquet"

WEATHER_FEATURE_NAMES = [
    "tmax_t0",
    "tmax_t1",
    "tmax_t3",
    "tmax_t7",
    "precip_t0",
    "precip_t1",
    "precip_t3",
    "precip_t7",
    "precip_sum_7d",
    "precip_sum_30d",
    "tmax_mean_7d",
    "days_since_rain",
    "kbdi",
    "kbdi_change_7d",
]


def load_weather_matrix(
    cell_ids: list[str] | None = None,
    weather_dir: Path | str | None = None,
    grid_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str], np.ndarray]:
    """Load complete cached weather files into (4383, n_cells) matrices.

    Skips any missing, incomplete, or invalid cells without forward-filling or interpolation.

    Returns:
        Tuple of (tmax_c, precip_mm, dates, cell_ids, has_weather_mask).
        Invalid cells contain NaNs in the matrix and False in has_weather_mask.
    """
    w_dir = Path(weather_dir or WEATHER_DIR)
    if cell_ids is None:
        if grid_df is None:
            grid_df = load_grid()
        cell_ids = grid_df["cell_id"].astype(str).tolist()

    dates = pd.date_range(start=WEATHER_START, end=WEATHER_END, freq="D")
    expected_dates = dates.date
    n_days = len(dates)
    assert n_days == WEATHER_N_DAYS

    n_cells = len(cell_ids)
    tmax_mat = np.full((n_days, n_cells), np.nan, dtype=np.float32)
    precip_mat = np.full((n_days, n_cells), np.nan, dtype=np.float32)
    has_weather = np.zeros(n_cells, dtype=bool)

    skipped_missing = 0
    skipped_invalid = 0

    for i, cid in enumerate(cell_ids):
        f_path = w_dir / f"{cid}.parquet"
        if not f_path.exists():
            skipped_missing += 1
            continue

        try:
            df = pd.read_parquet(f_path)
            if len(df) != WEATHER_N_DAYS:
                skipped_invalid += 1
                logger.warning("Cell %s skipped: expected %d rows, got %d", cid, WEATHER_N_DAYS, len(df))
                continue

            df_dates = pd.to_datetime(df["date"]).dt.date.values
            if not (df_dates == expected_dates).all():
                skipped_invalid += 1
                logger.warning("Cell %s skipped: date sequence mismatch", cid)
                continue

            tmax_vals = df["temperature_2m_max"].values
            precip_vals = df["precipitation_sum"].values

            if np.isnan(tmax_vals).any() or np.isnan(precip_vals).any():
                skipped_invalid += 1
                logger.warning("Cell %s skipped: contains NaN values", cid)
                continue

            if (tmax_vals < -20.0).any() or (tmax_vals > 60.0).any() or (precip_vals < 0.0).any():
                skipped_invalid += 1
                logger.warning("Cell %s skipped: values out of physical bounds", cid)
                continue

            tmax_mat[:, i] = tmax_vals.astype(np.float32)
            precip_mat[:, i] = precip_vals.astype(np.float32)
            has_weather[i] = True
        except Exception as e:
            skipped_invalid += 1
            logger.warning("Cell %s skipped due to read error: %s", cid, e)

    n_valid = int(has_weather.sum())
    print(
        f"Weather Matrix Loaded: {n_valid}/{n_cells} valid cells "
        f"(skipped: {skipped_missing} missing, {skipped_invalid} invalid)"
    )
    return tmax_mat, precip_mat, dates, list(cell_ids), has_weather


def mean_annual_rainfall_mm(
    precip_2d: np.ndarray,
    dates: pd.DatetimeIndex,
    cell_ids: list[str] | None = None,
    save_path: Path | str | None = None,
) -> np.ndarray:
    """Compute mean annual rainfall (mm) over training years (2015-2022) and register fit."""
    start_year, end_year = MEAN_RAIN_YEARS
    years = dates.year
    mask = (years >= start_year) & (years <= end_year)

    fit_start = dates[mask].min().date()
    fit_end = dates[mask].max().date()
    num_years = float(end_year - start_year + 1)

    register_fit(
        name="mean_annual_rainfall",
        first_date=fit_start,
        last_date=fit_end,
        metadata={"years": MEAN_RAIN_YEARS, "num_years": num_years, "allow_year_end": True},
    )

    total_precip = precip_2d[mask].sum(axis=0, dtype=np.float64)
    mean_annual = (total_precip / num_years).astype(np.float32)

    if save_path is not None and cell_ids is not None:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_rain = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "mean_annual_mm": mean_annual,
                "fit_start": str(fit_start),
                "fit_end": str(fit_end),
            }
        )
        df_rain.to_parquet(out_file, index=False)

    return mean_annual


def compute_days_since_rain(precip_2d: np.ndarray, threshold_mm: float = 2.5, cap: int = 365) -> np.ndarray:
    """Compute days since last rainfall strictly above threshold_mm, capped at 365.

    Returns 0 on a day where precip > threshold_mm. If no rain > threshold has
    occurred since day 0, returns (day_index + 1) capped at 365.
    """
    n_days, n_cells = precip_2d.shape
    dsr = np.zeros((n_days, n_cells), dtype=np.float32)
    current = np.zeros(n_cells, dtype=np.int32)

    for d in range(n_days):
        rained = precip_2d[d] > threshold_mm
        current = np.where(rained, 0, np.minimum(current + 1, cap))
        dsr[d] = current

    return dsr


def compute_weather_and_drought_features(
    tmax_c: np.ndarray,
    precip_mm: np.ndarray,
    mean_annual_mm: np.ndarray,
    dates: pd.DatetimeIndex,
    cell_ids: list[str],
    eval_start: str | date = TRAIN_START,
    eval_end: str | date = TEST_END,
) -> pd.DataFrame:
    """Compute all 14 weather and KBDI features on 2014-2025 matrices and slice to eval dates."""
    n_days, n_cells = tmax_c.shape
    assert precip_mm.shape == (n_days, n_cells)
    assert len(dates) == n_days
    assert len(cell_ids) == n_cells

    # Rolling sums via cumulative sums
    cs_p = np.cumsum(precip_mm.astype(np.float64), axis=0)
    p_sum_7d = cs_p.copy()
    if n_days > 7:
        p_sum_7d[7:] = cs_p[7:] - cs_p[:-7]

    p_sum_30d = cs_p.copy()
    if n_days > 30:
        p_sum_30d[30:] = cs_p[30:] - cs_p[:-30]

    cs_t = np.cumsum(tmax_c.astype(np.float64), axis=0)
    t_sum_7d = cs_t.copy()
    if n_days > 7:
        t_sum_7d[7:] = cs_t[7:] - cs_t[:-7]
    # Denominator is 7 for indices >= 6, and i+1 for i < 6
    denom_7d = np.minimum(np.arange(1, n_days + 1), 7).astype(np.float64)[:, None]
    t_mean_7d = t_sum_7d / denom_7d

    # Days since rain (> 2.5 mm)
    dsr = compute_days_since_rain(precip_mm, threshold_mm=2.5, cap=365)

    # KBDI using full 2014 spin-up matrix
    kbdi_mat = compute_kbdi(tmax_c, precip_mm, mean_annual_mm, q0=KBDI_Q0)

    # Target eval slice indices (all >= 2015-01-03, so t-7 is always >= 2014-12-27 inside 2014-2025 matrix)
    eval_s = pd.to_datetime(eval_start)
    eval_e = pd.to_datetime(eval_end)
    eval_mask = (dates >= eval_s) & (dates <= eval_e)
    eval_idx = np.where(eval_mask)[0]
    eval_dates = dates[eval_mask]
    n_eval = len(eval_dates)

    if len(eval_idx) > 0 and eval_idx[0] < 7:
        raise ValueError("Need at least 7 days of spin-up before eval_start for t-7 lags.")

    def _slice_cell_major(arr_2d: np.ndarray) -> np.ndarray:
        return arr_2d[eval_idx].T.flatten().astype(np.float32)

    data_dict = {
        "cell_id": pd.Series(np.repeat(cell_ids, n_eval), dtype="category"),
        "date": pd.Series(np.tile(eval_dates, n_cells), dtype="datetime64[ns]"),
        "tmax_t0": _slice_cell_major(tmax_c),
        "tmax_t1": tmax_c[eval_idx - 1].T.flatten().astype(np.float32),
        "tmax_t3": tmax_c[eval_idx - 3].T.flatten().astype(np.float32),
        "tmax_t7": tmax_c[eval_idx - 7].T.flatten().astype(np.float32),
        "precip_t0": _slice_cell_major(precip_mm),
        "precip_t1": precip_mm[eval_idx - 1].T.flatten().astype(np.float32),
        "precip_t3": precip_mm[eval_idx - 3].T.flatten().astype(np.float32),
        "precip_t7": precip_mm[eval_idx - 7].T.flatten().astype(np.float32),
        "precip_sum_7d": _slice_cell_major(p_sum_7d),
        "precip_sum_30d": _slice_cell_major(p_sum_30d),
        "tmax_mean_7d": _slice_cell_major(t_mean_7d),
        "days_since_rain": _slice_cell_major(dsr),
        "kbdi": _slice_cell_major(kbdi_mat),
        "kbdi_change_7d": (kbdi_mat[eval_idx] - kbdi_mat[eval_idx - 7]).T.flatten().astype(np.float32),
    }

    df_out = pd.DataFrame(data_dict)

    # Validation checks
    for col in WEATHER_FEATURE_NAMES:
        assert df_out[col].dtype == np.float32, f"{col} is not float32"
        assert df_out[col].isna().sum() == 0, f"{col} contains NaNs"
    assert (df_out["kbdi"] >= 0.0).all() and (df_out["kbdi"] <= 800.0).all(), "KBDI out of [0, 800]"

    return df_out


def build_partial_weather_features(output_path: Path | str | None = None) -> pd.DataFrame:
    """Build and save weather + KBDI features for all currently cached cells."""
    out_file = Path(output_path or FEATURES_WEATHER_PARTIAL_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tmax_all, precip_all, dates, cell_ids, has_weather = load_weather_matrix()
    valid_idx = np.where(has_weather)[0]
    valid_cells = [cell_ids[i] for i in valid_idx]

    if not valid_cells:
        raise RuntimeError("No valid cached weather cells found.")

    tmax_valid = tmax_all[:, valid_idx]
    precip_valid = precip_all[:, valid_idx]

    mean_rain = mean_annual_rainfall_mm(
        precip_valid,
        dates,
        cell_ids=valid_cells,
        save_path=MEAN_ANNUAL_RAIN_PATH,
    )

    df_feat = compute_weather_and_drought_features(
        tmax_c=tmax_valid,
        precip_mm=precip_valid,
        mean_annual_mm=mean_rain,
        dates=dates,
        cell_ids=valid_cells,
    )

    df_feat.to_parquet(out_file, index=False)
    print(f"Saved weather features for {len(valid_cells)} cells ({len(df_feat):,} rows) to {out_file}")
    print("\nFeature Summary (min / max / mean / NaNs):")
    for col in WEATHER_FEATURE_NAMES:
        s = df_feat[col]
        print(f"  {col:18s}: min={s.min():8.2f} | max={s.max():8.2f} | mean={s.mean():8.2f} | NaNs={s.isna().sum()}")

    return df_feat

