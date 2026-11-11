# Model v0 SHAP Sanity Audit

- **Model version:** `v0_plain_20261111` (`xgb_v0_plain`, 7 base features, no weather)
- **Route used:** `shap==0.54.1` (`shap.TreeExplainer`, exact TreeSHAP)
- **Units:** All SHAP values (`shap`) and base value are in **log-odds (margin) units**, NOT probabilities.
- **Expected (base) value:** `-3.189854` log-odds
- **Additivity check (`|sum(shap) + base - margin|`):** max diff `1.37e-05` (1,000 random rows), `1.62e-05` (20,000 global rows), `2.89e-06` (11,209 High/Critical local rows) — all <= `1e-4` (**PASS**).
- **Scope verified:** Strictly 2023 validation primary-set rows (`split == "val"`). Local parquet (`56,045` rows = `11,209` cell-days x top 5) computed in `18.10s`.

## 1. Global Feature Ranking (20,000 Random 2023 Primary Rows, Seed 42)

| Rank | Feature | Display Label | Mean \|SHAP\| (log-odds) | Share |
|---|---|---|---|---|
| 1 | `neighbour_fire_7d` | Fires in nearby cells, last 7 days | 0.643847 | 23.16% |
| 2 | `fire_count_30d` | Fires in this cell, last 30 days | 0.580108 | 20.87% |
| 3 | `lat_c` | Location | 0.551110 | 19.82% |
| 4 | `month_cos` | Time of year | 0.380177 | 13.68% |
| 5 | `month_sin` | Time of year | 0.306752 | 11.03% |
| 6 | `lon_c` | Location | 0.260779 | 9.38% |
| 7 | `fire_count_7d` | Fires in this cell, last 7 days | 0.057118 | 2.05% |

## 2. Do Recent Fires Increase Predicted Risk?

- `fire_count_7d` (Fires in this cell, last 7 days): mean SHAP when `> 0` is `+0.2773` (n=1,713) vs `== 0` `-0.0344` (n=18,287); Spearman `rho = +0.4834`.
- `fire_count_30d` (Fires in this cell, last 30 days): mean SHAP when `> 0` is `+0.5983` (n=6,240) vs `== 0` `-0.5717` (n=13,760); Spearman `rho = +0.8182`.
- `neighbour_fire_7d` (Fires in nearby cells, last 7 days): mean SHAP when `> 0` is `+0.2956` (n=8,929) vs `== 0` `-0.8684` (n=11,071); Spearman `rho = +0.9085`.

## 3. Location and Seasonality Dominance Check

- **Location (`lat_c` + `lon_c`) share:** `29.21%` (sum mean |SHAP| = `0.8119`)
- **Time of year (`month_sin` + `month_cos`) share:** `24.71%` (sum mean |SHAP| = `0.6869`)
- **Combined Location + Time-of-Year share:** `53.92%` vs **Recent Fire History share:** `46.08%`.

- **Finding:** Top rank is `neighbour_fire_7d` (`46.08%` fire-history share vs `53.92%` location + time-of-year share); static priors do not overwhelm recent fire activity.
