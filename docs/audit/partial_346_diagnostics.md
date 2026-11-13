# Diagnostics Audit on Partial Weather Cells (`partial: 346 cells`)

**Status:** PASS  
**Execution Mode:** `partial: 346 cells` (cached weather complete = 346 cells, 4,383 days)  
**Strict Safeguards Enforced:**  
- No hyperparameter tuning performed.
- No Model v1 winner selected.
- No calibration applied.
- Test split locked (`ALLOW_TEST_SPLIT` unset; 2024–2025 data untouched).

---

## 1. Uncertainty: Block Bootstrap by Calendar Week of 2023

1,000 block bootstrap resamples by calendar week of 2023 (seed 42) on the 2023 validation primary set (`110,484` cell-days across `53` weekly blocks, `5,669` positive fires). Evaluated for paired differences:
- `G2 - G1` (Weather contribution over fire/spatial/calendar baseline)
- `G3 - G2` (Drought/KBDI contribution over weather features)
- `G3 - climatology` (Full feature model over historical empirical baseline)

### Bootstrap 95% Confidence Intervals

| Contrast | Point Estimate Diff | Bootstrap Mean Diff | Bootstrap Std Err | 95% Bootstrap CI | Includes 0? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`G2 - G1`** | **+0.021343** | +0.020133 | 0.008278 | **[+0.005216, +0.036211]** | **No (False)** |
| **`G3 - G2`** | **-0.002122** | -0.002196 | 0.001825 | **[-0.005883, +0.001185]** | **Yes (True)** |
| **`G3 - climatology`** | **+0.048398** | +0.047547 | 0.008684 | **[+0.031273, +0.065312]** | **No (False)** |

### Findings on Uncertainty
1. **`G2 - G1` does NOT include 0:** The 95% interval `[+0.005216, +0.036211]` is strictly positive. Adding the 12 meteorological lag features provides a statistically significant improvement in PR-AUC over fire history and calendar baselines in 2023.
2. **`G3 - G2` DOES include 0:** The 95% interval `[-0.005883, +0.001185]` spans zero. The slight empirical difference (`-0.002122`) when adding KBDI on top of full weather features with default hyperparameters is not statistically distinguishable from zero.
3. **`G3 - climatology` does NOT include 0:** The 95% interval `[+0.031273, +0.065312]` is strictly positive and well above zero, confirming model skill over the historical baseline.

*Traceability:* `docs/audit/partial_346_bootstrap_summary.csv` and `docs/audit/partial_346_bootstrap.csv`.

---

## 2. Rolling-Origin Repeat (Validation Years 2021, 2022, 2023)

Evaluated across chronological rolling origins on the exact same 346 cells using the primary label `new_fire_next_48h` (`~burning_prev3`):
- **Validation Year 2021:** Train 2015–2019, Stage 1 early-stop 2020, Stage 2 refit 2015–2020 (`110,881` val cell-days, `5,022` positives, base rate `0.045292`).
- **Validation Year 2022:** Train 2015–2020, Stage 1 early-stop 2021, Stage 2 refit 2015–2021 (`111,493` val cell-days, `5,186` positives, base rate `0.046514`).
- **Validation Year 2023:** Train 2015–2021, Stage 1 early-stop 2022, Stage 2 refit 2015–2022 (`110,484` val cell-days, `5,669` positives, base rate `0.051311`).
*Years 2024–2025 remain strictly locked in the test split.*

### Results Table: Rolling-Origin Comparison

| Val Year | Group / Model | Features | Best Trees | PR-AUC | PR-AUC / Base | Recall@10% | Precision@5% | Brier |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2021** | `climatology` | 0 | 0 | 0.227260 | 5.018 | 0.4220 | 0.2174 | 0.038457 |
| **2021** | `G1` | 7 | 92 | 0.281502 | 6.215 | 0.4100 | 0.2392 | 0.036890 |
| **2021** | `G2` | 19 | 123 | 0.287538 | 6.349 | 0.4054 | 0.2443 | 0.036694 |
| **2021** | `G3` | 21 | 93 | 0.287871 | 6.356 | 0.4075 | 0.2461 | 0.036683 |
| **2022** | `climatology` | 0 | 0 | 0.226688 | 4.874 | 0.4108 | 0.2253 | 0.039635 |
| **2022** | `G1` | 7 | 137 | 0.259935 | 5.588 | 0.4111 | 0.2396 | 0.038577 |
| **2022** | `G2` | 19 | 144 | 0.262205 | 5.637 | 0.4200 | 0.2314 | 0.038537 |
| **2022** | `G3` | 21 | 157 | 0.263917 | 5.674 | 0.4135 | 0.2321 | 0.038493 |
| **2023** | `climatology` | 0 | 0 | 0.222830 | 4.343 | 0.3759 | 0.2399 | 0.043791 |
| **2023** | `G1` | 7 | 226 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 |
| **2023** | `G2` | 19 | 111 | 0.273351 | 5.327 | 0.4060 | 0.2602 | 0.042202 |
| **2023** | `G3` | 21 | 113 | 0.271228 | 5.286 | 0.4048 | 0.2544 | 0.042249 |

### Operational Capacity Metrics: G3 vs Climatology

| Year | Metric | Climatology | G3 | Difference (G3 - Climatology) |
| :---: | :--- | :---: | :---: | :---: |
| **2021** | Recall@10% | 0.4220 | 0.4075 | -0.0145 |
| **2021** | Precision@5% | 0.2174 | 0.2461 | +0.0287 |
| **2022** | Recall@10% | 0.4108 | 0.4135 | +0.0027 |
| **2022** | Precision@5% | 0.2253 | 0.2321 | +0.0068 |
| **2023** | Recall@10% | 0.3759 | 0.4048 | +0.0289 |
| **2023** | Precision@5% | 0.2399 | 0.2544 | +0.0145 |

Recall at 10% does not consistently beat climatology.

### Longitudinal Observations
1. **Weather Gain Across Validation Years:** The weather gain (G2 - G1 PR-AUC) was +0.006, +0.002 and +0.021 in 2021, 2022 and 2023 respectively, and only 2023 has an interval.
2. **Climatology Stability:** The empirical baseline PR-AUC remains consistent across years (`0.227260` in 2021, `0.226688` in 2022, `0.222830` in 2023).
3. **KBDI Value in Earlier Origins:** In both 2021 and 2022, `G3` shows minor positive differences over `G2` (`+0.000333` in 2021, `+0.001712` in 2022).

*Traceability:* `docs/audit/partial_346_rolling.csv`.

---

## 3. KBDI Check: Feature Group Ablation on 2023 Validation

Comparison of feature subsets on 2023 `val` primary set (`110,484` rows, `5,669` positives, base rate `0.051311`):
- **`G1` (7 features):** Fire history, calendar, and static coordinates.
- **`G1K` (9 features):** `G1` + KBDI features (`kbdi`, `kbdi_change_7d`) only (no daily weather).
- **`G2` (19 features):** `G1` + 12 weather features (including cumulative rainfall sums and `days_since_rain`).
- **`G2R` (18 features):** `G2` without rainfall accumulation (`precip_sum_7d`, `precip_sum_30d`, `days_since_rain`), but with KBDI features (`kbdi`, `kbdi_change_7d`).
- **`G3` (21 features):** Full feature set (`G2` + KBDI features).

### Results Table: KBDI Ablation

| Group | Features | Description | Best Trees | PR-AUC | PR-AUC / Base | Recall@10% | Precision@5% | Brier | Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `G1` | 7 | Baseline fire history, calendar, static | 226 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 | Benchmark |
| `G1K` | 9 | `G1` + KBDI only (no weather lags) | 225 | 0.257314 | 5.015 | 0.4089 | 0.2542 | 0.042678 | Single year, no interval, not tested |
| `G2` | 19 | `G1` + 12 weather lags (with rain sums) | 111 | 0.273351 | 5.327 | 0.4060 | 0.2602 | 0.042202 | Benchmark |
| `G2R` | 18 | `G2` minus rain sums/days_since_rain + KBDI | 329 | 0.275441 | 5.368 | 0.4032 | 0.2561 | 0.042153 | Single year, no interval, not tested |
| `G3` | 21 | Full feature set (`G2` + KBDI) | 113 | 0.271228 | 5.286 | 0.4048 | 0.2544 | 0.042249 | Benchmark |

### Diagnostic Observations
1. **G1K and G2R Comparison Status:** Differences for G1K and G2R are single year, no interval, not tested.
2. **G2R vs G2:** There is no detectable difference (G2R vs G2 +0.002, same size as the G3-G2 bootstrap standard error 0.0018).
3. **Tree Count Variation:** Best tree counts per group in 2023: G1 226, G2 111, G3 113, G2R 329. These tree counts add noise to the comparison.

*Traceability:* `docs/audit/partial_346_kbdi_ablation.csv`.

---

## 4. Fire-Season View: Evaluation on Peak Positive Months

### Selection of Peak Months from Training Data (2015–2022)
To avoid validation lookahead leakage, peak fire months were determined solely from the 2015–2022 training data on the 346 partial cells (`900,085` primary cell-days, `38,987` positive fires):

| Month | Training Cell-Days | Training Positive Fires | Monthly Prevalence | Share of All Fires | Peak Rank |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **March (3)** | 64,214 | 6,545 | 10.19% | 16.79% | **1** |
| **April (4)** | 63,896 | 5,803 | 9.08% | 14.88% | **2** |
| **February (2)** | 62,921 | 5,563 | 8.84% | 14.27% | **3** |
| January (1) | 73,110 | 5,079 | 6.95% | 13.03% | 4 |
| December (12) | 73,846 | 4,490 | 6.08% | 11.52% | 5 |
| May (5) | 73,934 | 3,883 | 5.25% | 9.96% | 6 |
| November (11) | 73,711 | 3,475 | 4.71% | 8.91% | 7 |
| October (10) | 81,480 | 1,730 | 2.12% | 4.44% | 8 |
| June (6) | 80,720 | 956 | 1.18% | 2.45% | 9 |
| September (9) | 82,049 | 604 | 0.74% | 1.55% | 10 |
| August (8) | 85,053 | 469 | 0.55% | 1.20% | 11 |
| July (7) | 85,151 | 392 | 0.46% | 1.01% | 12 |

**Chosen Peak Fire Season:** Months **2, 3, 4 (February, March, April)** hold **17,911 fires (45.94% of all training fires)** and are the only months with baseline fire prevalence exceeding 8.8%.  
*(We also evaluate the full pre-monsoon dry season months 2, 3, 4, 5, which hold 55.90% of all fires).*

### Performance Table on Peak Fire Months (2023 Validation)

#### Core Peak Season: Months 2–3–4 (February, March, April)
- Scope: `23,304` cell-days, `2,406` positives, base rate `0.103244` (10.32%).

| Group / Model | Features | PR-AUC | PR-AUC / Base | Recall@10% | Precision@5% | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `climatology` | 0 | 0.251412 | 2.435 | 0.3100 | 0.3839 | 0.085299 |
| `G1` | 7 | 0.281914 | 2.731 | 0.3442 | 0.3972 | 0.083269 |
| `G2` | 19 | 0.318873 | 3.089 | 0.3404 | 0.4275 | 0.080929 |
| `G3` | 21 | 0.316211 | 3.063 | 0.3469 | 0.4011 | 0.080987 |

#### Extended Pre-Monsoon Season: Months 2–3–4–5 (February through May)
- Scope: `32,742` cell-days, `2,975` positives, base rate `0.090862` (9.09%).

| Group / Model | Features | PR-AUC | PR-AUC / Base | Recall@10% | Precision@5% | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `climatology` | 0 | 0.233918 | 2.574 | 0.3020 | 0.3421 | 0.076487 |
| `G1` | 7 | 0.265558 | 2.923 | 0.3497 | 0.3569 | 0.074607 |
| `G2` | 19 | 0.297548 | 3.275 | 0.3416 | 0.3835 | 0.072839 |
| `G3` | 21 | 0.294279 | 3.239 | 0.3459 | 0.3584 | 0.072970 |

### Seasonal Observations
1. **Seasonal Prevalence and Baseline PR-AUC:** Baseline fire prevalence is roughly double the full-year rate (10.32% vs 5.13%), and PR-AUC is correspondingly higher across all models during peak months (`0.318873` for `G2`, `0.316211` for `G3`).
2. **Alert Concentration:** In the peak season, Precision@5% is `0.4275` (42.8%) for `G2` and `0.4011` (40.1%) for `G3`.

*Traceability:* `docs/audit/partial_346_season.csv`.

---

## 5. Data Traceability Matrix

Every metric reported in this document is traced directly to permanent audit CSV artifacts:

| Audit Section | Metric / Table | Source File Path |
| :--- | :--- | :--- |
| **Section 1** | Bootstrap draws (1,000 resamples) | [partial_346_bootstrap.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_346_bootstrap.csv) |
| **Section 1** | 95% Confidence Intervals & zero checks | [partial_346_bootstrap_summary.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_346_bootstrap_summary.csv) |
| **Section 2** | Rolling-origin 2021 & 2022 evaluations | [partial_346_rolling.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_346_rolling.csv) |
| **Section 3** | KBDI ablation (`G1`, `G1K`, `G2`, `G2R`, `G3`) | [partial_346_kbdi_ablation.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_346_kbdi_ablation.csv) |
| **Section 4** | Fire-season evaluations (months 2-3-4 & 2-3-4-5) | [partial_346_season.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_346_season.csv) |
| **All** | Baseline cell cohort (346 cells) | [partial_cells.json](file:///c:/Users/himan/Documents/Mini-Project/data/processed/partial_cells.json) |
