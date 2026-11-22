# Model v0 to Model v1 API & Dashboard Swap Checklist

This document audits the schemas, column data types, and interface contracts for swapping the operational pipeline from **Model v0** (`xgb_v0_plain`, 7 fire-history and static features) to **Model v1** (`xgb_v1_tuned_G3`, 21 features with comprehensive meteorology and drought dynamics).

---

## 1. Upstream Data Files Read by API & Dashboard

The API (`src/api/`) and Streamlit Dashboard (`src/dashboard/`) consume three primary spatial datasets:

1. **Spatial Grid Definition:** `data/processed/grid_cells_0.5deg.parquet`
2. **Model Predictions Table:** `data/processed/predictions/model_v1_dev.parquet` (replacing `model_v0_dev.parquet`)
3. **Local SHAP Explanations Table:** `data/processed/shap/model_v1_dev.parquet` (replacing `model_v0_dev.parquet`)
   *(Additionally, audit table `data/processed/shap/model_v1_dev_by_feature.parquet` mirrors `model_v0_dev_by_feature.parquet`)*

---

## 2. Parquet Schema & Dtype Comparison

### A. Spatial Grid (`data/processed/grid_cells_0.5deg.parquet`)
- **Status:** Unchanged across model versions (canonical 0.5-degree spatial discretization across India).
- **Row count:** 1,167 active terrestrial cells.
- **Columns & Dtypes:**
  - `cell_id`: string / object (e.g. `"r06c21"`)
  - `lat_c`: float64 / float32 (cell centroid latitude)
  - `lon_c`: float64 / float32 (cell centroid longitude)
  - `row`: int64 / int32
  - `col`: int64 / int32
  - `state`: string / object

### B. Predictions Table (`data/processed/predictions/model_v1_dev.parquet` vs `model_v0_dev.parquet`)
- **Evaluation Split:** 2023 validation primary set (`split == "val"`, `burning_prev3 == False`).
- **Row count:** Exactly 373,605 cell-days (1,043 primary-set active cells x 358 dates).

| Column Name | Model v0 Dtype | Model v1 Dtype | Compatibility | Notes |
|---|---|---|---|---|
| `cell_id` | `object` (str) | `object` (str) | **Identical** | Canonical cell identifier |
| `date` | `datetime64[ns]` | `datetime64[ns]` | **Identical** | Observation date in 2023 |
| `score` | `float32` | `float32` | **Identical** | Calibrated/raw probability in `[0, 1]` |
| `tier` | `category` | `category` | **Identical** | Discrete tiers: `Low`, `Moderate`, `High`, `Critical` |
| `model_version` | `object` (str) | `object` (str) | **Identical** | Version tag (see differences below) |
| `split` | `object` (str) | `object` (str) | **Identical** | `"val"` strictly |

### C. Grouped SHAP Explanations (`data/processed/shap/model_v1_dev.parquet` vs `model_v0_dev.parquet`)
- **Scope:** Every cell-day in tier `High` or `Critical` in 2023.
- **Row count:** Exactly 56,045 rows (11,209 cell-days x 5 ranks).

| Column Name | Model v0 Dtype | Model v1 Dtype | Compatibility | Notes |
|---|---|---|---|---|
| `cell_id` | `object` (str) | `object` (str) | **Identical** | Canonical cell identifier |
| `date` | `datetime64[ns]` | `datetime64[ns]` | **Identical** | Observation date in 2023 |
| `rank` | `int8` | `int8` | **Identical** | Integer rank `1` to `5` |
| `driver` | `object` (str) | `object` (str) | **Identical** | Name of the driver group |
| `shap` | `float32` | `float32` | **Identical** | Summed SHAP contribution in **log-odds** margin |
| `feature_detail` | `object` (str) | `object` (str) | **Identical** | Raw feature values formatted as `"f1=v1, f2=v2"` |
| `model_version` | `object` (str) | `object` (str) | **Identical** | Model version string |

### D. Audit Per-Feature SHAP (`data/processed/shap/model_v1_dev_by_feature.parquet` vs `model_v0_dev_by_feature.parquet`)
- **Row count:** Exactly 56,045 rows (11,209 cell-days x 5 ranks).

| Column Name | Model v0 Dtype | Model v1 Dtype | Compatibility | Notes |
|---|---|---|---|---|
| `cell_id` | `object` (str) | `object` (str) | **Identical** | Canonical cell identifier |
| `date` | `datetime64[ns]` | `datetime64[ns]` | **Identical** | Observation date in 2023 |
| `rank` | `int8` | `int8` | **Identical** | Integer rank `1` to `5` |
| `feature` | `object` (str) | `object` (str) | **Identical** | Name of the feature |
| `label` | `object` (str) | `object` (str) | **Identical** | Human-readable feature label |
| `value` | `float32` | `float32` | **Identical** | Feature value on that date |
| `shap` | `float32` | `float32` | **Identical** | Per-feature SHAP value in log-odds units |
| `model_version` | `object` (str) | `object` (str) | **Identical** | Model version string |

---

## 3. Key Differences for Dashboard & API Handlers

While schema and data types are 100% backwards-compatible, the dashboard and API must accommodate the following structural content changes:

1. **Model Version Identifier:**
   - Model v0: `"v0_plain_20261111"`
   - Model v1: `"v1_tuned_20261121"`
   - *Impact:* Any hardcoded filter on `model_version == "v0_plain_..."` must be generalized or updated to accept `v1_tuned_*`.

2. **Expanded Driver Groups (8 Groups vs 5 Groups):**
   - In Model v0, there were exactly 5 driver groups, so every cell-day had all 5 groups present in ranks 1 to 5.
   - In Model v1, there are **8 driver groups** spanning 21 features. For each cell-day, only the **top 5 groups** by $|summed\ SHAP|$ are preserved in `model_v1_dev.parquet`.
   - **Three new driver groups appear in v1:**
     - `"Recent heat"` (comprising `tmax_t0, tmax_t1, tmax_t3, tmax_t7, tmax_mean_7d`)
     - `"Recent rainfall"` (comprising `precip_t0, precip_t1, precip_t3, precip_t7, precip_sum_7d, precip_sum_30d, days_since_rain`)
     - `"Drought index"` (comprising `kbdi, kbdi_change_7d`)
   - *Impact:* Dashboard color palettes, icon mappings, and filter dropdowns must include the 3 new driver group names.

3. **Tier Threshold Shifts:**
   - Model v0 thresholds: P90 = `0.124508`, P97 = `0.252007`, P99.5 = `0.443128`.
   - Model v1 thresholds: P90 = `0.133950`, P97 = `0.273928`, P99.5 = `0.469649`.
   - *Impact:* Threshold configuration must read from `configs/tiers_v1_preview.yaml` (or `configs/tiers_preview.yaml`).

4. **Log-Odds Units Clarification:**
   - All SHAP attributions remain in **log-odds margin units**, exactly reproducing the booster margin via `base_value + sum(shap) == margin`. Dashboards presenting waterfall or bar plots must label units as "Log-Odds Margin Contribution", not "Probability Shift".
