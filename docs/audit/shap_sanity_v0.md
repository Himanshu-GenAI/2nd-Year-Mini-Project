# Model v0 SHAP Sanity Audit

- **Model version:** `v0_plain_20261111` (`xgb_v0_plain`, 7 base features grouped into 5 drivers)
- **Route used:** `shap==0.54.1` (`shap.TreeExplainer`, exact TreeSHAP)
- **Units:** All SHAP values (`shap`) and base value are in **log-odds (margin) units**, NOT probabilities.
- **Expected (base) value:** `-3.189854` log-odds
- **Grouped additivity check (`|sum_g(shap_g) + base - margin|`):** max diff `1.37e-05` (1,000 random rows), `1.62e-05` (20,000 global rows), `2.89e-06` (11,209 High/Critical local rows) — all <= `1e-4` (**PASS**).
- **Scope verified:** Strictly 2023 validation primary-set rows (`split == "val"`). Grouped local parquet (`56,045` rows = `11,209` cell-days x top 5 drivers) + audit per-feature parquet computed in `19.74s`.

## 1. Grouped Driver Ranking (`mean(|sum_g SHAP_g|)`, 20,000 Random 2023 Primary Rows, Seed 42)

| Rank | Driver (Grouped) | Mean \|Summed SHAP\| (log-odds) | Share |
|---|---|---|---|
| 1 | Regional baseline | 0.688811 | 27.51% |
| 2 | Fires in nearby cells, last 7 days | 0.643847 | 25.71% |
| 3 | Fires in this cell, last 30 days | 0.580108 | 23.17% |
| 4 | Seasonal baseline | 0.534284 | 21.34% |
| 5 | Fires in this cell, last 7 days | 0.057118 | 2.28% |

## 2. Per-Feature Audit Ranking (`docs/audit/shap_global_v0.csv`)

| Rank | Feature | Driver Group | Mean \|SHAP\| (log-odds) | Share |
|---|---|---|---|---|
| 1 | `neighbour_fire_7d` | Fires in nearby cells, last 7 days | 0.643847 | 23.16% |
| 2 | `fire_count_30d` | Fires in this cell, last 30 days | 0.580108 | 20.87% |
| 3 | `lat_c` | Regional baseline | 0.551110 | 19.82% |
| 4 | `month_cos` | Seasonal baseline | 0.380177 | 13.68% |
| 5 | `month_sin` | Seasonal baseline | 0.306752 | 11.03% |
| 6 | `lon_c` | Regional baseline | 0.260779 | 9.38% |
| 7 | `fire_count_7d` | Fires in this cell, last 7 days | 0.057118 | 2.05% |

## 3. Do Recent Fires Increase Predicted Risk?

- `fire_count_7d` (Fires in this cell, last 7 days): mean SHAP when `> 0` is `+0.2773` (n=1,713) vs `== 0` `-0.0344` (n=18,287); Spearman `rho = +0.4834`.
- `fire_count_30d` (Fires in this cell, last 30 days): mean SHAP when `> 0` is `+0.5983` (n=6,240) vs `== 0` `-0.5717` (n=13,760); Spearman `rho = +0.8182`.
- `neighbour_fire_7d` (Fires in nearby cells, last 7 days): mean SHAP when `> 0` is `+0.2956` (n=8,929) vs `== 0` `-0.8684` (n=11,071); Spearman `rho = +0.9085`.

## 4. Regional & Seasonal Baseline Dominance Check (Grouped)

- **Regional baseline (`lat_c + lon_c`) share:** `27.51%` (mean |summed SHAP| = `0.6888`)
- **Seasonal baseline (`month_sin + month_cos`) share:** `21.34%` (mean |summed SHAP| = `0.5343`)
- **Combined baseline share:** `48.84%` vs **Combined recent fire activity share:** `51.16%`.

> **FINDING (Static Regional/Seasonal Baseline Dominance in Grouped Model v0 SHAP):** `Regional baseline` ranks #1 at grouped driver level and combined `Regional baseline` + `Seasonal baseline` accounts for `48.84%` of grouped mean |SHAP| (`51.16%` from fire counts). Because `xgb_v0_plain` lacks dynamic weather and drought inputs, geographic coordinates (`lat_c + lon_c`) and calendar seasonality (`month_sin + month_cos`) act together as the model's largest single risk prior.
