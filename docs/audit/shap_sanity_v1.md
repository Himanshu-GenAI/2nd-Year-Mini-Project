# Model v1 SHAP Sanity Audit

- **Model version:** `v1_tuned_20261121` (`xgb_v1_tuned_G3`, 21 features grouped into 8 driver groups)
- **Route used:** `shap==0.54.1` (`shap.TreeExplainer`, exact TreeSHAP)
- **Units:** All SHAP values (`shap`) and base value are in **log-odds (margin) units**, NOT probabilities.
- **Expected (base) value:** `-3.162649` log-odds
- **Grouped additivity check (`|sum_g(shap_g) + base - margin|`):** max diff `1.42e-05` (1,000 random rows), `1.86e-05` (20,000 global rows), `3.20e-06` (11,209 High/Critical local rows) — all <= `1e-4` (**PASS**).
- **Scope verified:** Strictly 2023 validation primary-set rows (`split == "val"`). Grouped local parquet (`56,045` rows = `11,209` cell-days x top 5 drivers) + audit per-feature parquet computed in `144.98s`.

## 1. Grouped Driver Ranking (`mean(|sum_g SHAP_g|)`, 20,000 Random 2023 Primary Rows, Seed 42)

| Rank | Driver (Grouped) | Mean \|Summed SHAP\| (log-odds) | Share |
|---|---|---|---|
| 1 | Fires in this cell, last 30 days | 0.662598 | 24.88% |
| 2 | Fires in nearby cells, last 7 days | 0.462100 | 17.35% |
| 3 | Regional baseline | 0.446292 | 16.75% |
| 4 | Recent rainfall | 0.335280 | 12.59% |
| 5 | Seasonal baseline | 0.330336 | 12.40% |
| 6 | Recent heat | 0.183381 | 6.88% |
| 7 | Drought index | 0.164781 | 6.19% |
| 8 | Fires in this cell, last 7 days | 0.078872 | 2.96% |

## 2. Per-Feature Ranking (`docs/audit/shap_global_v1_features.csv`)

| Rank | Feature | Driver Group | Mean \|SHAP\| (log-odds) | Share |
|---|---|---|---|---|
| 1 | `fire_count_30d` | Fires in this cell, last 30 days | 0.662598 | 21.45% |
| 2 | `neighbour_fire_7d` | Fires in nearby cells, last 7 days | 0.462100 | 14.96% |
| 3 | `lat_c` | Regional baseline | 0.353483 | 11.44% |
| 4 | `month_cos` | Seasonal baseline | 0.248014 | 8.03% |
| 5 | `precip_t0` | Recent rainfall | 0.238912 | 7.73% |
| 6 | `month_sin` | Seasonal baseline | 0.177560 | 5.75% |
| 7 | `lon_c` | Regional baseline | 0.167103 | 5.41% |
| 8 | `kbdi` | Drought index | 0.141118 | 4.57% |
| 9 | `tmax_t0` | Recent heat | 0.138091 | 4.47% |
| 10 | `fire_count_7d` | Fires in this cell, last 7 days | 0.078872 | 2.55% |
| 11 | `precip_sum_30d` | Recent rainfall | 0.055328 | 1.79% |
| 12 | `days_since_rain` | Recent rainfall | 0.050982 | 1.65% |
| 13 | `kbdi_change_7d` | Drought index | 0.049150 | 1.59% |
| 14 | `precip_sum_7d` | Recent rainfall | 0.043366 | 1.40% |
| 15 | `tmax_mean_7d` | Recent heat | 0.042184 | 1.37% |
| 16 | `tmax_t7` | Recent heat | 0.040496 | 1.31% |
| 17 | `tmax_t1` | Recent heat | 0.039637 | 1.28% |
| 18 | `tmax_t3` | Recent heat | 0.033268 | 1.08% |
| 19 | `precip_t1` | Recent rainfall | 0.027269 | 0.88% |
| 20 | `precip_t7` | Recent rainfall | 0.020962 | 0.68% |
| 21 | `precip_t3` | Recent rainfall | 0.018738 | 0.61% |

## 3. Directional Sanity: Do Recent Fires Raise Predicted Risk?

- `fire_count_7d` -> **Fires in this cell, last 7 days**: mean group SHAP when `> 0` is `+0.2948` (n=1,713) vs `== 0` `-0.0585` (n=18,287); Spearman `rho = +0.4849` (**PASS**).
- `fire_count_30d` -> **Fires in this cell, last 30 days**: mean group SHAP when `> 0` is `+0.4988` (n=6,240) vs `== 0` `-0.7366` (n=13,760); Spearman `rho = +0.8195` (**PASS**).
- `neighbour_fire_7d` -> **Fires in nearby cells, last 7 days**: mean group SHAP when `> 0` is `+0.2400` (n=8,929) vs `== 0` `-0.5881` (n=11,071); Spearman `rho = +0.9076` (**PASS**).

## 4. Directional Sanity: Weather and Drought Physical Expectations

- `kbdi` (Higher KBDI should raise fire risk): Spearman `rho = +0.9131` (expected positive) -> **PASS**
- `tmax_t0` (Higher maximum temperature should raise fire risk): Spearman `rho = +0.8911` (expected positive) -> **PASS**
- `tmax_mean_7d` (Higher 7d mean max temperature should raise fire risk): Spearman `rho = -0.0682` (expected positive) -> **OPPOSITE SIGN**
- `precip_sum_7d` (Higher recent rainfall should lower fire risk): Spearman `rho = -0.2384` (expected negative) -> **PASS**
- `precip_sum_30d` (Higher 30d rainfall should lower fire risk): Spearman `rho = -0.4260` (expected negative) -> **PASS**
- `days_since_rain` (More days since rain should raise fire risk): Spearman `rho = +0.6768` (expected positive) -> **PASS**

### Physical Anomaly Reporting

- **Opposite-sign feature `tmax_mean_7d` (overall rho=-0.0682):** Monthly correlation breakdown: M1:+0.20, M2:+0.29, M3:+0.07, M4:-0.47, M5:-0.63, M6:-0.64, M7:+0.46, M8:+0.41, M9:+0.29, M10:+0.00, M11:+0.28, M12:+0.32.

## 5. Share of Total Grouped Absolute SHAP Across Clusters

- **Regional & Seasonal Baseline (`Regional baseline` + `Seasonal baseline`):** `29.16%` (mean |summed SHAP| = `0.7766`)
- **Fire Activity Groups (3 groups):** `45.19%` (mean |summed SHAP| = `1.2036`)
- **Weather & Drought Groups (`Recent heat` + `Recent rainfall` + `Drought index`):** `25.66%` (mean |summed SHAP| = `0.6834`)

> **SUMMARY:** In Model v1, dynamic environmental drivers (weather + drought + fire activity) account for `70.84%` of grouped attribution, reducing static geographic and calendar baseline dominance from v0's levels.
