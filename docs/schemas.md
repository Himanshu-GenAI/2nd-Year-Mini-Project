# Dataset Schemas & Interface Specifications

**Project:** Wildfire Early Alert System for India (PS-113)  
**Last Updated:** Week 2 (2026-11-09)

---

## 1. Parquet Datasets

### A. Clean Active Fire Telemetry (`data/processed/detections_clean.parquet`)
- **Row Count:** 811,280 detections (filtered from 923,584 raw MODIS observations)
- **Filters Applied:** `drop_duplicates(subset=[lat, lon, acq_date, acq_time, satellite])`, `type == 0` (presumed vegetation), `confidence >= 30`, spatial grid assignment to kept cells.
- **Columns:**
  | Column | Dtype | Description |
  |---|---|---|
  | `latitude` | `float64` | Observation latitude |
  | `longitude` | `float64` | Observation longitude |
  | `acq_date` | `object` (`datetime.date`) | UTC acquisition date |
  | `acq_time` | `int32` | UTC acquisition time (HHMM) |
  | `satellite` | `string` | MODIS platform (`Terra` or `Aqua`) |
  | `confidence` | `int32` | Detection confidence percentage (30–100) |
  | `frp` | `float32` | Fire Radiative Power (MW) |
  | `cell_id` | `category` / `str` | Spatial grid identifier (`r{row:02d}c{col:02d}`) |

---

### B. Fire Cell-Day Panel (`data/processed/fire_cell_day.parquet`)
- **Row Count:** 4,684,338 cell-days (1,167 cells × 4,014 dates from 2015-01-03 to 2025-12-29)
- **Trimmed Dates:** First 2 days (2015-01-01, 2015-01-02; insufficient 3-day history) and last 2 days (2025-12-30, 2025-12-31; missing 48h forward label) are dropped.
- **Split Distribution:**
  - `train`: 3,405,306 rows (2015-01-03 to 2022-12-29)
  - `val`: 423,621 rows (2023-01-01 to 2023-12-29)
  - `test`: 850,743 rows (2024-01-01 to 2025-12-29)
  - `gap`: 4,668 rows (2022-12-30, 2022-12-31, 2023-12-30, 2023-12-31)
- **Columns:**
  | Column | Dtype | Description |
  |---|---|---|
  | `cell_id` | `category` | Cell identifier (`r00c00` to `r63c61`) |
  | `date` | `datetime64[ns]` | Date of observation day $t$ |
  | `n_det` | `uint16` | Count of active fire detections in cell on day $t$ |
  | `frp_sum` | `float32` | Sum of Fire Radiative Power (MW) on day $t$ |
  | `burning_prev3` | `bool` | True if cell had $\ge 1$ detection on days $t-2, t-1, t$ |
  | `fire_next_48h` | `int8` | 1 if cell has $\ge 1$ detection on $t+1$ or $t+2$; else 0 |
  | `new_fire_next_48h` | `int8` | 1 if `fire_next_48h == 1` AND `burning_prev3 == False`; else 0 |
  | `split` | `category` | Partition name (`train`, `val`, `test`, `gap`) |

---

### C. Weather Cell Files (`data/raw/weather/cells/{cell_id}.parquet`)
- **Row Count:** 4,383 daily records per cell (continuous 2014-01-01 to 2025-12-31)
- **Columns:**
  | Column | Dtype | Description |
  |---|---|---|
  | `date` | `object` (`datetime.date`) | Calendar date (GMT / UTC) |
  | `temperature_2m_max` | `float32` | Maximum 2-meter air temperature (°C) |
  | `precipitation_sum` | `float32` | Daily accumulated precipitation (mm) |
  | `cell_id` | `str` | Grid cell identifier |
  | `fetched_at` | `str` | UTC ISO-8601 timestamp of harvest |

---

## 2. Primary Evaluation Set Rule

Operational wildfire early alert systems prioritize predicting **new ignitions** rather than continuous burns.
- **Rule:** For the primary target label `new_fire_next_48h`, the primary evaluation set strictly excludes cells that are already actively burning in the lookback window.
- **Implementation:** `primary_mask(df) = ~df["burning_prev3"]`.
- When evaluating `fire_next_48h` (all active burns), all cell-days are retained (`set_name = "all"`).

---

## 3. Core Module Function Signatures

### `src/data/labels.py`
- `primary_mask(df: pd.DataFrame) -> pd.Series`
- `build_full_grid_date_index() -> tuple[list[str], pd.DatetimeIndex]`
- `assign_split(dates: pd.Series) -> pd.Series`
- `build_fire_cell_day_table(detections_df=None, output_path=None) -> pd.DataFrame`
- `compute_and_save_label_stats(df: pd.DataFrame) -> pd.DataFrame`

### `src/models/evaluate.py`
- `base_rate(df: pd.DataFrame, label: str) -> float`
- `pr_auc(df: pd.DataFrame, label: str, score: str) -> float`
- `recall_at_top_frac(df, label, score, frac=0.10, cell_col='cell_id', date_col='date') -> tuple[float, int]`
- `precision_at_top_frac(df, label, score, frac=0.05, cell_col='cell_id', date_col='date') -> float`
- `brier(df: pd.DataFrame, label: str, score: str) -> float`
- `reliability_table(df, label, score, bins=10) -> pd.DataFrame`
- `plot_reliability(table: pd.DataFrame, path: Path | str) -> None`
- `evaluate(df, label, score, model, split, set_name='primary') -> dict[str, Any]`
- `record_result(result: dict[str, Any], csv_path=None, md_path=None) -> None`

### `src/features/kbdi.py`
- `celsius_to_fahrenheit(celsius: float | np.ndarray) -> float | np.ndarray`
- `mm_to_inches(mm: float | np.ndarray) -> float | np.ndarray`
- `mean_annual_rainfall_mm(weather_df, cell_col='cell_id', date_col='date', precip_col='precipitation_sum') -> pd.Series`
- `compute_kbdi(tmax_c, precip_mm, mean_annual_mm, q0=0.0) -> np.ndarray`
- `compute_kbdi_for_study_period(tmax_c, precip_mm, mean_annual_mm, dates, q0=0.0) -> tuple[np.ndarray, pd.DatetimeIndex]`
