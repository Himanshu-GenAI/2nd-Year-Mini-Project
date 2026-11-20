# Full-Cell Run Diagnostics (1,167 cells)

**Date:** 2026-11-21
**Grid Cells:** 1,167 (100% complete weather cache)
**Status:** Full run complete, test split locked (2024-2025 untouched)

## 1. Ablation Results (2023 Validation, Primary Set)

| Group | Params | Features | PR-AUC | Recall@10% | Prec@5% | Brier | Trees | Time (s) |
|---|---|---|---|---|---|---|---|---|
| climatology | N/A | 0 | 0.219835 | 0.3770 | 0.2260 | 0.042091 | 0 | 0.0 |
| G1 | default | 7 | 0.247788 | 0.4257 | 0.2473 | 0.041241 | 393 | 105.0 |
| G1K | default | 9 | 0.256559 | 0.4121 | 0.2456 | 0.041006 | 294 | 75.4 |
| G2 | default | 19 | 0.274038 | 0.4259 | 0.2495 | 0.040475 | 442 | 129.9 |
| G2R | default | 18 | 0.273256 | 0.4189 | 0.2497 | 0.040504 | 404 | 128.1 |
| G3 | default | 21 | 0.273958 | 0.4200 | 0.2526 | 0.040479 | 472 | 146.4 |
| G3 | tuned | 21 | 0.278369 | 0.4328 | 0.2544 | 0.040351 | 455 | 192.9 |

## 2. Sensitivity Check (Fixed 150 Trees, No Early Stopping)

| Group | Trees | PR-AUC | Recall@10% | Prec@5% | Brier |
|---|---|---|---|---|---|
| climatology | 0 | 0.219835 | 0.3770 | 0.2260 | 0.042091 |
| G1 | 150 | 0.243507 | 0.4124 | 0.2424 | 0.041361 |
| G1K | 150 | 0.253406 | 0.4100 | 0.2432 | 0.041096 |
| G2 | 150 | 0.271202 | 0.4154 | 0.2470 | 0.040598 |
| G2R | 150 | 0.272047 | 0.4108 | 0.2451 | 0.040586 |
| G3 | 150 | 0.271582 | 0.4127 | 0.2465 | 0.040587 |

## 3. Block Bootstrap Uncertainty (1,000 Calendar-Week Resamples, Seed 42)

| Contrast | Point Diff | 95% CI Lower | 95% CI Upper | Includes 0? | Verdict |
|---|---|---|---|---|---|
| G2 - G1 | +0.026250 | +0.009379 | +0.042168 | No | significant |
| G3 - G2 | -0.000080 | -0.002353 | +0.002122 | Yes | inconclusive |
| G1K - G1 | +0.008771 | +0.001489 | +0.013939 | No | significant |
| G2R - G2 | -0.000782 | -0.003928 | +0.002706 | Yes | inconclusive |
| G3 - climatology | +0.054123 | +0.041200 | +0.066695 | No | significant |
| G3 - G1 | +0.026170 | +0.009470 | +0.042485 | No | significant |

## 4. Rolling-Origin Multi-Year Evaluation

| Val Year | Group | Features | PR-AUC | Recall@10% | Prec@5% | Brier | Trees |
|---|---|---|---|---|---|---|---|
| 2021 | climatology | 0 | 0.235573 | 0.4071 | 0.2105 | 0.037629 | 0 |
| 2021 | G1 | 7 | 0.280258 | 0.4172 | 0.2257 | 0.036386 | 255 |
| 2021 | G2 | 19 | 0.293501 | 0.4316 | 0.2321 | 0.036048 | 210 |
| 2021 | G3 | 21 | 0.292617 | 0.4359 | 0.2298 | 0.036048 | 181 |
| 2022 | climatology | 0 | 0.211115 | 0.4053 | 0.2054 | 0.037710 | 0 |
| 2022 | G1 | 7 | 0.247494 | 0.4104 | 0.2226 | 0.036631 | 247 |
| 2022 | G2 | 19 | 0.253130 | 0.4262 | 0.2275 | 0.036427 | 389 |
| 2022 | G3 | 21 | 0.249524 | 0.4245 | 0.2212 | 0.036519 | 368 |

## 5. Benchmarks & Secondary Labels

- **Random Forest (G3, 200 trees):** PR-AUC: 0.266698, Recall@10%: 0.4115, Brier: 0.122538
- **Secondary Label (`fire_next_48h`, all rows):** PR-AUC: 0.556722, Brier: 0.056681

## 6. Model v1 Winner Selection (Decision D18)

- **Winning Model:** `xgb_v1_G3_tuned`
- **Feature Group:** `G3` (21 features)
- **2023 Primary PR-AUC:** `0.278369`
- **Recall@10%:** `0.4328`

## 7. Feature Importance (Permutation PR-AUC & Gain)

| Rank | Feature | Perm Mean | Perm Std | Gain |
|---|---|---|---|---|
| 1 | `fire_count_30d` | 0.083739 | 0.002028 | 141.0876 |
| 2 | `precip_t0` | 0.040618 | 0.001880 | 29.1914 |
| 3 | `neighbour_fire_7d` | 0.036621 | 0.001283 | 97.7440 |
| 4 | `lat_c` | 0.034794 | 0.000527 | 18.4683 |
| 5 | `lon_c` | 0.026287 | 0.000652 | 12.4628 |
| 6 | `month_cos` | 0.021126 | 0.001201 | 23.9416 |
| 7 | `fire_count_7d` | 0.015306 | 0.000506 | 67.4537 |
| 8 | `tmax_t0` | 0.013474 | 0.000626 | 6.3957 |
| 9 | `month_sin` | 0.011546 | 0.000301 | 33.0538 |
| 10 | `kbdi_change_7d` | 0.005623 | 0.000321 | 6.1371 |
| 11 | `precip_t1` | 0.005385 | 0.000148 | 4.8785 |
| 12 | `kbdi` | 0.004481 | 0.000333 | 12.7830 |
| 13 | `tmax_t7` | 0.003324 | 0.000325 | 5.8661 |
| 14 | `tmax_mean_7d` | 0.002567 | 0.000114 | 7.4669 |
| 15 | `tmax_t3` | 0.002001 | 0.000286 | 5.2013 |

---
*All numbers traced to saved CSV files in `docs/audit/`.*