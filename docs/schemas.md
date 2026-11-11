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

### D. Model v0 Development Predictions (`data/processed/predictions/model_v0_dev.parquet`)
- **Row Count:** 373,605 cell-days (2023 validation primary evaluation set: `split == "val"` and `burning_prev3 == False`). Never contains training or test split rows.
- **Scope Note (Primary-Set Only vs Replay):** `model_v0_dev.parquet` contains **only primary-set rows** (cells actively burning in `t-2..t`, i.e., `burning_prev3 == True`, are excluded). The **real replay prediction file** produced for operational simulation/UI must include **all cell-days** with an explicit `burning_prev3` boolean flag.
- **Tier Thresholds Config:** `configs/tiers_v0_preview.yaml` (preview only, computed on 2023 primary-set raw scores: `P90 = 0.136186`, `P97 = 0.273916`, `P99.5 = 0.463278`).
- **Columns:**
  | Column | Dtype | Description |
  |---|---|---|
  | `cell_id` | `str` | Grid cell identifier (`r{row:02d}c{col:02d}`) |
  | `date` | `datetime64[ns]` | Validation date in 2023 (`2023-01-01` to `2023-12-29`) |
  | `score` | `float32` | Raw uncalibrated predicted probability of `new_fire_next_48h` |
  | `tier` | `category` | Risk tier (`Low` < P90, `Moderate` P90–P97, `High` P97–P99.5, `Critical` >= P99.5) |
  | `model_version` | `str` | Version identifier (`v0_plain_20261111`) |
  | `split` | `str` | Data split (`"val"` only) |

---

### E. Model v0 Grouped Local SHAP Explanations (`data/processed/shap/model_v0_dev.parquet`)
- **Scope:** Top-5 **driver groups** by `|summed shap|` (`GROUPS` in `src/explain/labels.py`) for every 2023 validation primary-set cell-day in tier `High` or `Critical` (`11,209` cell-days × `min(5, 5 groups)` = `56,045` rows). Never contains train or test rows.
- **Grouping & Units:** Feature SHAP attributions are computed with `shap.TreeExplainer` and summed within each display group (`Regional baseline` = `lat_c + lon_c`; `Seasonal baseline` = `month_sin + month_cos`; fire-count features remain individual drivers). All `shap` values are summed attributions in **log-odds (model margin) units**, NOT probabilities (`expected_value + sum_g(shap_g) == XGBoost output_margin`).
- **Columns:**
  | Column | Dtype | Description |
  |---|---|---|
  | `cell_id` | `str` | Grid cell identifier |
  | `date` | `datetime64[ns]` | Validation date in 2023 (`2023-01-01` to `2023-12-29`) |
  | `rank` | `int8` | Group importance rank within `(cell_id, date)` sorted descending by `|shap|` (`1` to `5`) |
  | `driver` | `str` | Grouped driver label from `GROUPS` (`src/explain/labels.py`); unique within `(cell_id, date)` |
  | `shap` | `float32` | Summed TreeSHAP attribution for the driver group in log-odds margin units |
  | `feature_detail` | `str` | Underlying feature(s) and values in the group (e.g., `lat_c=23.25, lon_c=79.75`) |
  | `model_version` | `str` | Model version matching `model_v0_dev.parquet` (`v0_plain_20261111`) |

---

### F. Model v0 Per-Feature Local SHAP Audit (`data/processed/shap/model_v0_dev_by_feature.parquet`)
- **Scope:** Audit-only per-feature top-5 attributions prior to driver grouping (`11,209` High/Critical cell-days × `5` = `56,045` rows), in log-odds margin units.
- **Columns:** `cell_id`, `date`, `rank` (`1..5`), `feature`, `label` (`FEATURE_LABELS`), `value`, `shap`, `model_version`.

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
