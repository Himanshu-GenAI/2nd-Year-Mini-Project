"""Central configuration for Wildfire Risk Forecasting system.

Defines study period, grid dimensions, temporal split boundaries,
FIRMS thresholds, weather quotas, KBDI parameters, and filesystem paths.
"""

from pathlib import Path

# Paths Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FIRMS_DIR = RAW_DIR / "firms"
RAW_WEATHER_DIR = RAW_DIR / "weather"
RAW_BOUNDARIES_DIR = RAW_DIR / "boundaries"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_AUDIT_DIR = PROCESSED_DIR / "audit"

DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_AUDIT_DIR = DOCS_DIR / "audit"

# File Paths
FIRMS_CSV_PATH = RAW_FIRMS_DIR / "fire_archive_M-C61_808434.csv"
BOUNDARY_PATH = RAW_BOUNDARIES_DIR / "Admin2.shp"
GRID_PATH = PROCESSED_DIR / "grid_cells_0.5deg.parquet"
DETECTIONS_CLEAN_PATH = PROCESSED_DIR / "detections_clean.parquet"
FIRE_TABLE_PATH = PROCESSED_DIR / "fire_cell_day.parquet"
WEATHER_DIR = RAW_WEATHER_DIR / "cells"
LEDGER_PATH = RAW_WEATHER_DIR / "quota_ledger.json"

# Study Period & Spin-up
STUDY_START = "2015-01-01"
STUDY_END = "2025-12-31"
KBDI_SPINUP_START = "2014-01-01"

# Spatial Grid (0.5 degree)
GRID_DEG = 0.5
LAT_MIN = 6.0
LAT_MAX = 38.0
LON_MIN = 67.0
LON_MAX = 98.0
# Candidate cells: ((LAT_MAX - LAT_MIN) / GRID_DEG) * ((LON_MAX - LON_MIN) / GRID_DEG) = 64 * 62 = 3,968

# Temporal Horizon and Lookback
LABEL_PREV_DAYS = 3  # days t-2, t-1, t (inclusive of t)
LABEL_HORIZON_DAYS = 2  # days t+1, t+2
# Day is defined by UTC acq_date

# Splits (sample dates t, inclusive)
TRAIN_START = "2015-01-03"  # first date with full 3-day history
TRAIN_END = "2022-12-29"
VAL_START = "2023-01-01"
VAL_END = "2023-12-29"
TEST_START = "2024-01-01"
TEST_END = "2025-12-29"
# 2-day gap days (30 and 31 Dec) belong to no split

# FIRMS Detection Thresholds
CONF_MIN = 30
FIRE_TYPE_KEEP = 0

# Weather API & Quota Settings
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_START = "2014-01-01"
WEATHER_END = "2025-12-31"
WEATHER_N_DAYS = 4383
WEATHER_VARS = ("temperature_2m_max", "precipitation_sum")
QUOTA_MINUTE_CAP = 480
QUOTA_HOUR_CAP = 4000
QUOTA_DAY_CAP = 8000
CELL_WEIGHT = 4383 / 14 * 2 / 10
MIN_INTERVAL_S = 55
SHUFFLE_SEED = 42

# Drought Index (KBDI) Parameters
RAIN_DAY_MM = 0.254
RAIN_SPELL_FIRST_IN = 0.20
KBDI_Q0 = 0
MEAN_RAIN_YEARS = (2015, 2022)
