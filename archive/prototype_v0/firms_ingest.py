"""NASA FIRMS Active Fire Archive Ingestion and Aggregation.

Ingests MODIS / VIIRS active fire archives, filters low-confidence detections,
projects point observations onto the spatial grid mesh, and aggregates daily
fire intensity and frequency statistics per grid cell.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    CONFIDENCE_THRESHOLD,
    FRP_MIN_MW,
    GRID_SIZE_DEG,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    RAW_FIRMS_DIR,
)
from src.data.grid import coords_to_grid_id


def load_and_preprocess_firms(
    file_path: str | Path | None = None,
    confidence_min: int = CONFIDENCE_THRESHOLD,
    frp_min: float = FRP_MIN_MW,
    clip_to_bounds: bool = True,
    sample_frac: float | None = None,
) -> pd.DataFrame:
    """Load NASA FIRMS CSV archive and apply validation and filtering.

    Args:
        file_path: Path to FIRMS CSV. If None, finds first CSV in RAW_FIRMS_DIR.
        confidence_min: Minimum MODIS detection confidence (0-100%).
        frp_min: Minimum Fire Radiative Power (MW).
        clip_to_bounds: Whether to filter within configured LAT/LON boundaries.
        sample_frac: Optional fraction to downsample for quick experimentation.

    Returns:
        Cleaned DataFrame of fire detections with assigned 'grid_id'.
    """
    if file_path is None:
        csv_files = list(RAW_FIRMS_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV archives found in {RAW_FIRMS_DIR}")
        file_path = csv_files[0]

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FIRMS archive not found at {file_path}")

    # Optimized dtypes to minimize memory usage on large NASA archives
    dtypes = {
        "latitude": "float32",
        "longitude": "float32",
        "brightness": "float32",
        "scan": "float32",
        "track": "float32",
        "satellite": "category",
        "instrument": "category",
        "confidence": "float32",
        "bright_t31": "float32",
        "frp": "float32",
        "daynight": "category",
    }

    df = pd.read_csv(file_path, dtype=dtypes, parse_dates=["acq_date"])

    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    # 1. Filter by detection confidence
    df = df[df["confidence"] >= confidence_min].copy()

    # 2. Filter by FRP threshold
    if frp_min > 0.0:
        df = df[df["frp"] >= frp_min]

    # 3. Spatial bounding box filter
    if clip_to_bounds:
        df = df[
            (df["latitude"] >= LAT_MIN)
            & (df["latitude"] <= LAT_MAX)
            & (df["longitude"] >= LON_MIN)
            & (df["longitude"] <= LON_MAX)
        ]

    # 4. Map point coordinates to discrete spatial grid cell ID
    df["grid_id"] = coords_to_grid_id(
        df["latitude"].values,
        df["longitude"].values,
        grid_size=GRID_SIZE_DEG,
    )

    return df


def aggregate_firms_daily(
    firms_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate individual fire detections into daily grid cell metrics.

    Args:
        firms_df: Preprocessed DataFrame containing 'grid_id' and 'acq_date'.

    Returns:
        DataFrame indexed by (grid_id, acq_date) with aggregated fire statistics:
        - fire_count: number of satellite fire detections in cell
        - max_frp: maximum FRP observed (MW)
        - sum_frp: sum of FRP observed
        - max_brightness: highest brightness temperature (K)
        - mean_brightness: mean brightness temperature (K)
        - day_fires: daytime detections count
        - night_fires: nighttime detections count
    """
    # Create indicators for day/night
    firms_df["is_day"] = (firms_df["daynight"] == "D").astype(int)
    firms_df["is_night"] = (firms_df["daynight"] == "N").astype(int)

    agg_spec = {
        "latitude": "count",
        "frp": ["max", "sum"],
        "brightness": ["max", "mean"],
        "is_day": "sum",
        "is_night": "sum",
    }

    grouped = firms_df.groupby(["grid_id", "acq_date"]).agg(agg_spec)
    grouped.columns = [
        "fire_count",
        "max_frp",
        "sum_frp",
        "max_brightness",
        "mean_brightness",
        "day_fires",
        "night_fires",
    ]

    grouped = grouped.reset_index()
    grouped["year"] = grouped["acq_date"].dt.year
    return grouped
