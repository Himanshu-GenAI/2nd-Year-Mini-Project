"""Central configuration for Wildfire Risk Forecasting system (Week 1).

Defines study period, grid dimensions, temporal split boundaries,
FIRMS thresholds, and filesystem paths.
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
