"""Meteorological Data Fetching and Generation Module.

Retrieves daily weather observations (temperature, precipitation, relative humidity,
wind speed) for spatial grid cells. Supports Open-Meteo Historical Weather API,
local parquet caching, and robust deterministic offline fallback generation for testing.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import RAW_WEATHER_DIR
from src.data.grid import grid_id_to_coords

logger = logging.getLogger(__name__)


def fetch_open_meteo_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timeout: int = 15,
) -> pd.DataFrame | None:
    """Retrieve daily meteorological timeseries from Open-Meteo API.

    Open-Meteo provides free open-access historical weather reanalysis (ERA5).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
        ],
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            daily = data.get("daily", {})
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(daily.get("time")),
                    "temp_max_c": daily.get("temperature_2m_max"),
                    "temp_min_c": daily.get("temperature_2m_min"),
                    "precip_24h_mm": daily.get("precipitation_sum"),
                    "relative_humidity_pct": daily.get("relative_humidity_2m_mean"),
                    "wind_speed_kmh": daily.get("wind_speed_10m_max"),
                }
            )
            return df
        else:
            logger.warning(
                f"Open-Meteo responded with status {response.status_code}: {response.text}"
            )
            return None
    except Exception as e:
        logger.warning(f"Failed to query Open-Meteo API: {e}. Falling back to cached/synthetic.")
        return None


def generate_synthetic_weather_timeseries(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic, physically consistent meteorological timeseries.

    Used when network access is unavailable, in CI pipelines, or for offline dev.
    Includes seasonal sinusoidal temperature cycle, monsoon precipitation pulse,
    and relative humidity inversely proportional to temperature.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    rng = np.random.default_rng(
        seed=random_seed + int(abs(latitude * 100) + abs(longitude * 100))
    )

    day_of_year = dates.dayofyear.values

    # Seasonal temperature cycle (peak in May/June for South Asia / Northern Hemisphere)
    temp_seasonal_base = 28.0 - (latitude - 20.0) * 0.4
    temp_annual_amp = 8.0
    # Peak near day 140-160 (May/June)
    temp_mean = temp_seasonal_base + temp_annual_amp * np.cos(
        2 * np.pi * (day_of_year - 150) / 365.25
    )
    temp_max = temp_mean + rng.normal(5.0, 1.5, size=n)
    temp_min = temp_mean - rng.normal(5.0, 1.2, size=n)

    # Monsoon precipitation model (strong pulse July-September, dry in spring)
    monsoon_envelope = np.clip(
        np.sin(np.pi * (day_of_year - 160) / 100), 0.0, 1.0
    ) ** 2
    rain_probability = 0.08 + 0.45 * monsoon_envelope
    rain_mask = rng.random(size=n) < rain_probability
    rain_amounts = np.zeros(n)
    rain_amounts[rain_mask] = rng.exponential(scale=18.0, size=np.sum(rain_mask))

    # Relative humidity: inversely related to temperature, positively related to rain
    rh = 55.0 - (temp_max - 28.0) * 1.5 + np.clip(rain_amounts * 3.0, 0, 35)
    rh += rng.normal(0, 5, size=n)
    rh = np.clip(rh, 15.0, 98.0)

    # Wind speed
    wind = 12.0 + rng.gamma(shape=2.5, scale=2.5, size=n)

    return pd.DataFrame(
        {
            "date": dates,
            "temp_max_c": np.round(temp_max, 2),
            "temp_min_c": np.round(temp_min, 2),
            "precip_24h_mm": np.round(rain_amounts, 2),
            "relative_humidity_pct": np.round(rh, 1),
            "wind_speed_kmh": np.round(wind, 1),
        }
    )


def fetch_weather_for_grid(
    grid_id: str,
    start_date: str,
    end_date: str,
    cache_dir: Path = RAW_WEATHER_DIR,
    force_refresh: bool = False,
    allow_synthetic_fallback: bool = True,
) -> pd.DataFrame:
    """Fetch or load cached daily weather for a specific grid cell.

    Args:
        grid_id: Spatial grid cell identifier.
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        cache_dir: Directory where weather parquets are saved.
        force_refresh: If True, bypasses local cache.
        allow_synthetic_fallback: If API fails, generate realistic data.

    Returns:
        DataFrame of daily meteorological variables with 'grid_id' and 'date'.
    """
    cache_path = cache_dir / f"weather_{grid_id}_{start_date[:4]}_{end_date[:4]}.parquet"

    if not force_refresh and cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    lat, lon = grid_id_to_coords(grid_id)

    # Attempt remote fetch
    df = fetch_open_meteo_weather(lat, lon, start_date, end_date)

    if df is None:
        if allow_synthetic_fallback:
            df = generate_synthetic_weather_timeseries(lat, lon, start_date, end_date)
        else:
            raise RuntimeError(f"Could not fetch weather data for grid {grid_id}")

    df["grid_id"] = grid_id

    # Cache locally
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        logger.warning(f"Could not save weather cache to {cache_path}: {e}")

    return df
