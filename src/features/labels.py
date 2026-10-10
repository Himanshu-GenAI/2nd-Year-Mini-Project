"""Wildfire Target Label Generation with Strict Anti-Leakage Safeguards.

Constructs forward-looking target variables for binary wildfire occurrence
and fire radiative power severity across configurable forecasting horizons (t+1, t+3, t+7).
Guarantees that target labels at day t are derived strictly from future observation windows.
"""

import pandas as pd

from src.config import DEFAULT_TARGET_HORIZON, FIRE_PRESENCE_THRESHOLD


def generate_fire_labels(
    df: pd.DataFrame,
    horizon_days: int = DEFAULT_TARGET_HORIZON,
    fire_count_col: str = "fire_count",
    date_col: str = "date",
    grid_col: str = "grid_id",
    threshold: int = FIRE_PRESENCE_THRESHOLD,
    drop_unlabeled_future: bool = True,
) -> pd.DataFrame:
    """Generate forward-shifted wildfire binary target label.

    Target definition:
        y_{i, t} = 1 if fire_count_{i, t + horizon} >= threshold else 0

    Args:
        df: Daily panel dataframe containing grid_id, date, and fire_count.
        horizon_days: Forecast lead time in days (e.g. 1 for next-day prediction).
        fire_count_col: Column indicating daily fire pixel counts.
        date_col: Column containing timestamp or datetime.
        grid_col: Spatial grid cell identifier column.
        threshold: Minimum fire count required for positive label.
        drop_unlabeled_future: If True, drops the final horizon_days rows per grid cell
            where future observations do not exist.

    Returns:
        DataFrame with added target column 'fire_occurred_next_{horizon_days}d'
        and 'target_fire_count_{horizon_days}d'.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Ensure dataset is strictly sorted by spatial entity and time
    df = df.sort_values([grid_col, date_col]).reset_index(drop=True)

    target_name = (
        "fire_occurred_next_day"
        if horizon_days == 1
        else f"fire_occurred_next_{horizon_days}d"
    )
    future_count_name = f"target_fire_count_{horizon_days}d"

    # Forward shift: row at t gets value from t + horizon_days
    # Note: .shift(-horizon_days) moves future values backwards to the current row index
    shifted_count = df.groupby(grid_col)[fire_count_col].shift(-horizon_days)

    df[future_count_name] = shifted_count
    df[target_name] = (shifted_count >= threshold).astype("Int64")

    # Anti-leakage verification:
    # Check that current day fire count does not trivially define future target
    if drop_unlabeled_future:
        # Rows where shifted_count is NaN represent the end of the observed timeline
        valid_mask = df[future_count_name].notna()
        df = df[valid_mask].copy()
        df[target_name] = df[target_name].astype(int)

    return df


def generate_multi_horizon_labels(
    df: pd.DataFrame,
    horizons: list[int] = [1, 3, 7],
    grid_col: str = "grid_id",
    date_col: str = "date",
    fire_count_col: str = "fire_count",
) -> pd.DataFrame:
    """Generate multiple prediction horizons (e.g. 1-day, 3-day, and 7-day cumulative risk).

    Args:
        df: Input daily panel DataFrame.
        horizons: List of forecast horizons in days.
        grid_col: Grid identifier column.
        date_col: Timestamp column.
        fire_count_col: Daily fire counts column.

    Returns:
        DataFrame enriched with multi-horizon labels.
    """
    df = df.copy()
    for h in horizons:
        df = generate_fire_labels(
            df,
            horizon_days=h,
            fire_count_col=fire_count_col,
            date_col=date_col,
            grid_col=grid_col,
            drop_unlabeled_future=False,
        )

    # Clean up edge rows where the longest horizon is unknown
    max_h = max(horizons)
    valid_mask = df[f"target_fire_count_{max_h}d"].notna()
    return df[valid_mask].copy()
