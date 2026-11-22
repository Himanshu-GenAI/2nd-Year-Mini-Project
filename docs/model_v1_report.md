# Model v1 Evaluation Report (2023 Validation Split)

Generated strictly from saved artifacts under `docs/` and `configs/`. Test split (`2024-2025`) remains locked.

> **Partial run tags present:** partial: 346 cells

> [!IMPORTANT]
> **Tuning Optimism Caveat:** The tuned hyperparameter configuration (Trial 04) was selected as the best of 20 random-search trials evaluated directly on the 2023 validation primary set. Consequently, 2023 validation scores are optimistic estimates of future generalization.

> [!NOTE]
> **Stage 1 vs. Refit Bundle Distinction:** The reported Model v1 benchmark metrics (`0.278369` PR-AUC, `0.4328` Recall@10%, `0.040351` Brier) are from the final Stage 2 refit booster saved in `models/model_v1/` (trained on the full 2015–2022 primary training set, 455 trees). The Stage 1 trial-04 model (trained on 2015–2021 with early-stopping on 2022) achieved `0.274267` PR-AUC on 2023.

## 1. Benchmark Comparison (`docs/results.csv`)

| model | split | label | set_name | rows | positives | base_rate | pr_auc | pr_auc_over_base | recall_top10 | precision_top5 | brier | dates_used | eval_date |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| climatology | val | fire_next_48h | all | 423621 | 38641 | 0.091216 | 0.427219 | 4.684 | 0.4285 | 0.3916 | 0.064422 | 363 | 2026-11-09 |
| climatology | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.219835 | 4.461 | 0.377 | 0.226 | 0.042091 | 358 | 2026-11-09 |
| logreg_v0 | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.216672 | 4.397 | 0.3539 | 0.222 | 0.042607 | 358 | 2026-11-10 |
| xgb_v0_plain | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.247788 | 5.029 | 0.4257 | 0.2473 | 0.041241 | 358 | 2026-11-10 |
| xgb_v0_spw | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.243413 | 4.94 | 0.4196 | 0.2449 | 0.162568 | 358 | 2026-11-10 |
| xgb_v0_sub20 | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.245515 | 4.982 | 0.4252 | 0.2453 | 0.06385 | 358 | 2026-11-10 |
| logreg_v0 | val | fire_next_48h | all | 423621 | 38641 | 0.091216 | 0.49664 | 5.445 | 0.4375 | 0.4298 | 0.06077 | 363 | 2026-11-10 |
| xgb_v0_plain | val | fire_next_48h | all | 423621 | 38641 | 0.091216 | 0.526162 | 5.768 | 0.4867 | 0.4486 | 0.058667 | 363 | 2026-11-10 |
| xgb_v0_spw | val | fire_next_48h | all | 423621 | 38641 | 0.091216 | 0.526047 | 5.767 | 0.4875 | 0.4481 | 0.134051 | 363 | 2026-11-10 |
| xgb_v0_sub20 | val | fire_next_48h | all | 423621 | 38641 | 0.091216 | 0.522622 | 5.73 | 0.4872 | 0.4467 | 0.089538 | 363 | 2026-11-10 |
| climatology | val | new_fire_next_48h | partial: 346 cells | 110484 | 5669 | 0.051311 | 0.22283 | 4.343 | 0.3759 | 0.2399 | 0.043791 | 348 | 2026-11-12 |
| xgb_v0_plain | val | new_fire_next_48h | partial: 346 cells | 110484 | 5669 | 0.051311 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 | 348 | 2026-11-12 |
| xgb_v1_G1 | val | new_fire_next_48h | partial: 346 cells | 110484 | 5669 | 0.051311 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 | 348 | 2026-11-12 |
| xgb_v1_G2 | val | new_fire_next_48h | partial: 346 cells | 110484 | 5669 | 0.051311 | 0.273351 | 5.327 | 0.406 | 0.2602 | 0.042202 | 348 | 2026-11-12 |
| xgb_v1_G3 | val | new_fire_next_48h | partial: 346 cells | 110484 | 5669 | 0.051311 | 0.271228 | 5.286 | 0.4048 | 0.2544 | 0.042249 | 348 | 2026-11-12 |
| xgb_v1_tuned | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.278369 | 5.649 | 0.4328 | 0.2544 | 0.040351 | 358 | 2026-11-20 |
| climatology | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.219835 | 4.461 | 0.377 | 0.226 | 0.042091 | 358 | 2026-11-20 |
| xgb_v1_G1_default | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.247788 | 5.029 | 0.4257 | 0.2473 | 0.041241 | 358 | 2026-11-20 |
| xgb_v1_G1K_default | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.256559 | 5.207 | 0.4121 | 0.2456 | 0.041006 | 358 | 2026-11-20 |
| xgb_v1_G2_default | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.274038 | 5.561 | 0.4259 | 0.2495 | 0.040475 | 358 | 2026-11-21 |
| xgb_v1_G2R_default | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.273256 | 5.545 | 0.4189 | 0.2497 | 0.040504 | 358 | 2026-11-21 |
| xgb_v1_G3_default | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.273958 | 5.56 | 0.42 | 0.2526 | 0.040479 | 358 | 2026-11-21 |
| xgb_v1_G3_tuned | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.278369 | 5.649 | 0.4328 | 0.2544 | 0.040351 | 358 | 2026-11-21 |
| rf_v1 | val | new_fire_next_48h | primary | 373605 | 18410 | 0.049277 | 0.266698 | 5.412 | 0.4115 | 0.2472 | 0.122538 | 358 | 2026-11-21 |
| xgb_v1_winner_all_rows | val | fire_next_48h | all | 423621 | 38641 | 0.091216 | 0.556722 | 6.103 | 0.4919 | 0.4588 | 0.056681 | 363 | 2026-11-21 |

## 2. Feature Group Ablation (`docs/audit/ablation.csv`)

| group | n_features | label | set_name | base_rate | pr_auc | pr_auc_over_base | recall_top10 | precision_top5 | brier | best_trees | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| climatology | 0 | new_fire_next_48h | partial: 346 cells | 0.051311 | 0.22283 | 4.343 | 0.3759 | 0.2399 | 0.043791 | 0 | 0.0 |
| xgb_v0_plain | 7 | new_fire_next_48h | partial: 346 cells | 0.051311 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 | 226 | 17.72 |
| G1 | 7 | new_fire_next_48h | partial: 346 cells | 0.051311 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 | 226 | 16.86 |
| G2 | 19 | new_fire_next_48h | partial: 346 cells | 0.051311 | 0.273351 | 5.327 | 0.406 | 0.2602 | 0.042202 | 111 | 11.63 |
| G3 | 21 | new_fire_next_48h | partial: 346 cells | 0.051311 | 0.271228 | 5.286 | 0.4048 | 0.2544 | 0.042249 | 113 | 13.04 |

## 3. Bootstrap Uncertainty Contrasts

Block bootstrap by calendar week of 2023 (1,000 resamples, seed 42) evaluated on the 2023 validation primary set.
All intervals are explicitly labelled with their hyperparameter settings (Default vs. Trial 04 tuned settings).

### Table 3A: Default Settings Refit Models (`docs/audit/full_run_bootstrap_summary.csv`)

| contrast | point_diff | boot_mean | boot_std | ci_lower_95 | ci_upper_95 | includes_zero | verdict |
|---|---|---|---|---|---|---|---|
| G2 - G1 | 0.02625 | 0.025151 | 0.008901 | 0.009379 | 0.042168 | False | significant |
| G3 - G2 | -8e-05 | -8.4e-05 | 0.001146 | -0.002353 | 0.002122 | True | inconclusive |
| G1K - G1 | 0.008771 | 0.008245 | 0.003201 | 0.001489 | 0.013939 | False | significant |
| G2R - G2 | -0.000782 | -0.000765 | 0.001694 | -0.003928 | 0.002706 | True | inconclusive |
| G3 - climatology | 0.054123 | 0.053361 | 0.006614 | 0.0412 | 0.066695 | False | significant |
| G3 - G1 | 0.02617 | 0.025067 | 0.008948 | 0.00947 | 0.042485 | False | significant |

### Table 3B: Trial 04 Tuned Settings Refit Models (`docs/audit/full_run_bootstrap_summary_trial04.csv`)

| contrast | point_diff | boot_mean | boot_std | ci_lower_95 | ci_upper_95 | includes_zero | verdict |
|---|---|---|---|---|---|---|---|
| G2 - G1 | 0.025923 | 0.024908 | 0.008625 | 0.008738 | 0.041734 | False | significant |
| G3 - G2 | 0.001686 | 0.001609 | 0.000994 | -0.000441 | 0.003486 | True | inconclusive |
| G1K - G1 | 0.008403 | 0.00797 | 0.003313 | 0.00111 | 0.014308 | False | significant |
| G2R - G2 | 0.002136 | 0.002114 | 0.000949 | 0.000357 | 0.003895 | False | significant |
| G3 - climatology | 0.058534 | 0.057711 | 0.006929 | 0.044751 | 0.071835 | False | significant |
| G3 - G1 | 0.027609 | 0.026517 | 0.008885 | 0.010209 | 0.043791 | False | significant |

### Side-by-Side Comparison: Default vs. Trial 04 Settings

| contrast | default_point_diff | default_95_ci | default_zero | trial04_point_diff | trial04_95_ci | trial04_zero |
|---|---|---|---|---|---|---|
| G2 - G1 | 0.02625 | [+0.009379, +0.042168] | Excludes 0 | 0.025923 | [+0.008738, +0.041734] | Excludes 0 |
| G3 - G2 | -8e-05 | [-0.002353, +0.002122] | Includes 0 | 0.001686 | [-0.000441, +0.003486] | Includes 0 |
| G1K - G1 | 0.008771 | [+0.001489, +0.013939] | Excludes 0 | 0.008403 | [+0.001110, +0.014308] | Excludes 0 |
| G2R - G2 | -0.000782 | [-0.003928, +0.002706] | Includes 0 | 0.002136 | [+0.000357, +0.003895] | Excludes 0 |
| G3 - climatology | 0.054123 | [+0.041200, +0.066695] | Excludes 0 | 0.058534 | [+0.044751, +0.071835] | Excludes 0 |
| G3 - G1 | 0.02617 | [+0.009470, +0.042485] | Excludes 0 | 0.027609 | [+0.010209, +0.043791] | Excludes 0 |

### Physical & Statistical Interpretations

- **Weather gain real?** **YES**. Under both Default (`[+0.009379, +0.042168]`) and Trial 04 (`[+0.008738, +0.041734]`) settings, the 95% bootstrap confidence interval for `G2 - G1` strictly excludes 0. Adding the 12 weather features provides a statistically confirmed gain of ~`+0.026` PR-AUC over fire history alone.
- **KBDI on top of weather inconclusive?** **YES**. Under both Default (`[-0.002353, +0.002122]`) and Trial 04 (`[-0.000441, +0.003486]`) settings, the 95% bootstrap confidence interval for `G3 - G2` spans across 0. The incremental benefit of adding KBDI drought features to the full weather suite is statistically inconclusive.
- **`G3 minus climatology`:** Under Default settings, the refit point difference is `+0.054123` with 95% CI `[+0.041200, +0.066695]`. Under Trial 04 tuned settings, the refit point difference is `+0.058534` (winner `0.278369` minus climatology `0.219835`, bootstrap mean `+0.057711`) with 95% CI `[+0.044751, +0.071835]`. Both strictly exclude 0 (statistically **significant**).
- **`G2R minus G2`:** Default refit point difference is `-0.000782` with 95% CI `[-0.003928, +0.002706]` (inconclusive). Trial 04 refit point difference is `+0.002136` with 95% CI `[+0.000357, +0.003895]` (weakly excludes 0).

## 4. Top Features (`docs/audit/feature_importance.csv`)

| feature | gain_importance | perm_importance_mean | perm_importance_std |
|---|---|---|---|
| fire_count_30d | 141.08758544921875 | 0.0837388849495746 | 0.0020284480015646 |
| precip_t0 | 29.191436767578125 | 0.0406180405560125 | 0.0018795588937593 |
| neighbour_fire_7d | 97.7440185546875 | 0.0366209487087307 | 0.0012825004428642 |
| lat_c | 18.46825218200684 | 0.034793846870613 | 0.0005267455988868 |
| lon_c | 12.462835311889648 | 0.0262873338208287 | 0.0006524422619667 |
| month_cos | 23.94160270690918 | 0.0211261234995469 | 0.0012010410574034 |
| fire_count_7d | 67.45374298095703 | 0.0153055113038165 | 0.000506202339331 |
| tmax_t0 | 6.395707607269287 | 0.0134736723371665 | 0.0006255895136612 |
| month_sin | 33.05380630493164 | 0.0115458562905319 | 0.0003005141671183 |
| kbdi_change_7d | 6.1370930671691895 | 0.0056229258812992 | 0.0003207990519885 |

## 5. Slice Breakdown (`docs/audit/model_v1_breakdown.csv`)

| dimension | bucket | rows | positives | base_rate | pr_auc | recall_top10 | brier | note |
|---|---|---|---|---|---|---|---|---|
| season | Winter (Dec-Feb) | 83147 | 7312 | 0.087941 | 0.34587 | 0.3994 | 0.067955 | ok |
| season | Pre-monsoon (Mar-May) | 86707 | 6782 | 0.078217 | 0.261892 | 0.357 | 0.064624 | ok |
| season | Monsoon (Jun-Sep) | 139569 | 1405 | 0.010067 | 0.095414 | 0.5319 | 0.009505 | ok |
| season | Post-monsoon (Oct-Nov) | 64182 | 2911 | 0.045355 | 0.23137 | 0.405 | 0.038877 | ok |
| state | Maharashtra | 31394 | 2337 | 0.074441 | 0.340934 | 0.2811 | 0.057224 | ok |
| state | Madhya Pradesh | 34080 | 2014 | 0.059096 | 0.274049 | 0.2983 | 0.048266 | ok |
| state | Uttar Pradesh | 29334 | 1756 | 0.059862 | 0.23556 | 0.339 | 0.050482 | ok |
| state | Karnataka | 19244 | 1481 | 0.076959 | 0.367171 | 0.2349 | 0.058415 | ok |
| state | Odisha | 15778 | 1158 | 0.073393 | 0.336195 | 0.4025 | 0.056569 | ok |
| state | Andhra Pradesh | 16976 | 1138 | 0.067036 | 0.259405 | 0.2346 | 0.055811 | ok |
| state | Gujarat | 22231 | 1057 | 0.047546 | 0.190831 | 0.2632 | 0.041906 | ok |
| state | Telangana | 11241 | 992 | 0.088248 | 0.26034 | 0.2131 | 0.07236 | ok |
| state | Chhattisgarh | 13283 | 855 | 0.064368 | 0.322037 | 0.4293 | 0.050599 | ok |
| state | Tamil Nadu | 15109 | 779 | 0.051559 | 0.18086 | 0.2393 | 0.045706 | ok |
| previous_activity | burning_prev3=False | 373605 | 18410 | 0.049277 | 0.278369 | 0.4328 | 0.040351 | ok |

## 6. Tier Preview Thresholds (`configs/tiers_preview.yaml`)

- **Model version:** `v1_tuned_20261121`
- **Date:** `2026-11-21`
- **Set:** `primary`
- **Cutoffs:** P90 = `0.13395016193389903`, P97 = `0.27392757177352917`, P99.5 = `0.4696485477685901`
- **Observed 2023 shares:** Low: 89.9999%, Moderate: 6.9999%, High: 2.5000%, Critical: 0.5003%
