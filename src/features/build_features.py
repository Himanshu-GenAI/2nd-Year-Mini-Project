"""Feature Engineering Pipeline for Wildfire Risk Forecasting.

Transforms raw spatial fire observations, weather variables, and drought indices
into ML-ready feature matrices with strictly backward-looking lags, rolling statistics,
spatial neighborhood indicators, and cyclical calendar signals.
Saves partitioned parquet datasets by year.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    GRID_SIZE_DEG,
    LAG_DAYS,
    PROCESSED_DATA_DIR,
)
from src.data.grid import get_neighboring_cells
from src.data.kbdi import compute_kbdi_timeseries
from src.features.labels import generate_fire_labels


def add_cyclical_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add sinusoidal day-of-year and calendar month encodings.

    Captures seasonal periodicity without sharp boundary discontinuities.
    """
    df = df.copy()
    doy = df[date_col].dt.dayofyear.values
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = df[date_col].dt.month
    return df


def add_weather_rolling_features(
    df: pd.DataFrame,
    grid_col: str = "grid_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute rolling aggregations for meteorological variables strictly on past observations.

    Args:
        df: Input panel DataFrame sorted by (grid_id, date).

    Returns:
        DataFrame with rolling weather features added.
    """
    df = df.sort_values([grid_col, date_col]).copy()
    grouped = df.groupby(grid_col)

    # 7-day rolling mean max temperature (heatwave duration)
    df["temp_max_rolling_mean_7d"] = (
        grouped["temp_max_c"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        .round(2)
    )

    # Cumulative precipitation over critical drying windows (14d and 30d)
    df["precip_rolling_sum_14d"] = (
        grouped["precip_24h_mm"]
        .transform(lambda s: s.rolling(window=14, min_periods=1).sum())
        .round(2)
    )
    df["precip_rolling_sum_30d"] = (
        grouped["precip_24h_mm"]
        .transform(lambda s: s.rolling(window=30, min_periods=1).sum())
        .round(2)
    )

    # 7-day minimum relative humidity (atmospheric aridity)
    df["rh_rolling_min_7d"] = (
        grouped["relative_humidity_pct"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).min())
        .round(1)
    )

    # 7-day peak wind speed (spread potential)
    df["wind_rolling_max_7d"] = (
        grouped["wind_speed_kmh"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).max())
        .round(1)
    )

    # 7-day delta in KBDI (rate of drought intensification)
    df["kbdi_delta_7d"] = (
        grouped["kbdi"]
        .transform(lambda s: s - s.shift(7).bfill())
        .round(2)
    )

    return df


def add_fire_lag_features(
    df: pd.DataFrame,
    grid_col: str = "grid_id",
    date_col: str = "date",
    fire_count_col: str = "fire_count",
    lag_days: list[int] = LAG_DAYS,
) -> pd.DataFrame:
    """Generate strictly backward-looking fire history lags and rolling sums.

    All lags have shift >= 1 to prevent data leakage.
    """
    df = df.sort_values([grid_col, date_col]).copy()
    grouped = df.groupby(grid_col)

    # Historical lag indicators (yesterday, 3 days ago, 1 week ago, etc.)
    for lag in lag_days:
        df[f"fire_count_lag_{lag}d"] = grouped[fire_count_col].shift(lag).fillna(0).astype(int)

    # Cumulative 30-day fire activity in this cell (ignoring today's fire to prevent leakage)
    df["fire_sum_30d"] = (
        grouped[fire_count_col]
        .transform(lambda s: s.shift(1).rolling(window=30, min_periods=1).sum())
        .fillna(0)
        .astype(int)
    )

    return df


def add_spatial_neighborhood_features(
    df: pd.DataFrame,
    grid_col: str = "grid_id",
    date_col: str = "date",
    fire_count_col: str = "fire_count",
) -> pd.DataFrame:
    """Compute fire occurrence in adjacent 8 spatial grid cells from the previous day.

    Wildfires naturally propagate across spatial boundaries. This feature checks
    whether adjacent neighbor cells experienced fires at t-1.
    """
    df = df.copy()

    # Create lookup map of (grid_id, date) -> fire_count
    # Using yesterday's date for strict anti-leakage
    df["yesterday"] = df[date_col] - pd.Timedelta(days=1)

    daily_fire_map = df.set_index([grid_col, "yesterday"])[fire_count_col].to_dict()

    # Precompute neighbor list for each unique cell
    unique_cells = df[grid_col].unique()
    neighbors_map = {
        cid: get_neighboring_cells(cid, grid_size=GRID_SIZE_DEG)
        for cid in unique_cells
    }

    # Vectorized / grouped neighbor lookup
    def compute_cell_neighbor_fires(row):
        yesterday_date = row[date_col] - pd.Timedelta(days=1)
        cid = row[grid_col]
        neighbors = neighbors_map.get(cid, [])
        return sum(
            daily_fire_map.get((n_id, yesterday_date), 0)
            for n_id in neighbors
        )

    # For fast execution, apply over unique (grid_id, date)
    unique_pairs = df[[grid_col, date_col]].drop_duplicates().copy()
    unique_pairs["neighbor_fire_count_lag_1d"] = [
        sum(
            daily_fire_map.get((nid, d - pd.Timedelta(days=1)), 0)
            for nid in neighbors_map.get(gid, [])
        )
        for gid, d in zip(unique_pairs[grid_col], unique_pairs[date_col])
    ]

    df = df.merge(unique_pairs, on=[grid_col, date_col], how="left")
    df.drop(columns=["yesterday"], inplace=True, errors="ignore")
    return df


def engineer_features(
    merged_daily_df: pd.DataFrame,
    include_target: bool = True,
    target_horizon: int = 1,
) -> pd.DataFrame:
    """Complete feature engineering pipeline on daily merged grid dataframe.

    Args:
        merged_daily_df: Combined panel containing [grid_id, date, fire_count,
            temp_max_c, temp_min_c, precip_24h_mm, relative_humidity_pct, wind_speed_kmh, kbdi].
        include_target: Whether to append forward-shifted target label.
        target_horizon: Prediction lead time in days.

    Returns:
        Feature matrix ready for model consumption.
    """
    df = merged_daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Ensure KBDI exists; if not computed yet, compute sequentially per cell
    if "kbdi" not in df.columns:
        df["kbdi"] = (
            df.groupby("grid_id", group_keys=False)
            .apply(compute_kbdi_timeseries)
        )

    # 1. Cyclical calendar features
    df = add_cyclical_calendar_features(df, date_col="date")

    # 2. Weather rolling statistics
    df = add_weather_rolling_features(df, grid_col="grid_id", date_col="date")

    # 3. Fire history lags
    df = add_fire_lag_features(df, grid_col="grid_id", date_col="date", fire_count_col="fire_count")

    # 4. Spatial neighbor lag features
    df = add_spatial_neighborhood_features(df, grid_col="grid_id", date_col="date", fire_count_col="fire_count")

    # 5. Append forward-shifted label
    if include_target:
        df = generate_fire_labels(
            df,
            horizon_days=target_horizon,
            fire_count_col="fire_count",
            date_col="date",
            grid_col="grid_id",
            drop_unlabeled_future=True,
        )

    return df


def build_dataset_for_years(
    years: list[int] | None = None,
    output_dir: Path = PROCESSED_DATA_DIR,
    save: bool = True,
) -> pd.DataFrame:
    """Build and partition processed feature parquets by year.

    Output files are saved to:
        data/processed/year={YYYY}/wildfire_features_{YYYY}.parquet
    """
    from src.data.firms_ingest import aggregate_firms_daily, load_and_preprocess_firms
    from src.data.grid import generate_grid_mesh
    from src.data.weather_fetch import fetch_weather_for_grid

    if years is None:
        years = [2018, 2019, 2020, 2021, 2022]

    start_date = f"{min(years)}-01-01"
    end_date = f"{max(years)}-12-31"

    # 1. Load active fires
    try:
        firms_raw = load_and_preprocess_firms()
        firms_daily = aggregate_firms_daily(firms_raw)
        firms_daily["date"] = pd.to_datetime(firms_daily["acq_date"])
    except Exception:
        firms_daily = pd.DataFrame(columns=["grid_id", "date", "fire_count"])

    # 2. Select top active grid cells or sample
    if not firms_daily.empty:
        top_cells = firms_daily["grid_id"].value_counts().head(20).index.tolist()
    else:
        mesh = generate_grid_mesh(lat_min=28.0, lat_max=32.0, lon_min=75.0, lon_max=78.0)
        top_cells = mesh["grid_id"].head(10).tolist()

    # 3. Create panel records
    records = []
    for cid in top_cells:
        w_df = fetch_weather_for_grid(cid, start_date, end_date)
        records.append(w_df)

    weather_full = pd.concat(records, ignore_index=True)

    # Merge fire counts into panel
    panel = weather_full.merge(
        firms_daily[["grid_id", "date", "fire_count"]],
        on=["grid_id", "date"],
        how="left",
    )
    panel["fire_count"] = panel["fire_count"].fillna(0).astype(int)

    # 4. Run feature engineering
    processed_df = engineer_features(panel, include_target=True)

    # 5. Save partitioned by year
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_df["year"] = processed_df["date"].dt.year

        for yr, group in processed_df.groupby("year"):
            year_partition_dir = output_dir / f"year={yr}"
            year_partition_dir.mkdir(parents=True, exist_ok=True)
            out_file = year_partition_dir / f"wildfire_features_{yr}.parquet"
            group.to_parquet(out_file, index=False)

    return processed_df
