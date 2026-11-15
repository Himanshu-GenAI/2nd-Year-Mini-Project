# Diagnostics Audit on Partial Weather Cells (`partial: 727 cells`)

**Status:** PASS  
**Execution Mode:** `partial: 727 cells` (cached weather complete = 727 cells, 4,383 days)  
**Strict Safeguards Enforced:**  
- No hyperparameter tuning performed.
- No Model v1 winner selected.
- No calibration applied.
- Test split locked (`ALLOW_TEST_SPLIT` unset; 2024–2025 data untouched).
- 346-cell results strictly preserved (no overwriting).

---

## 1. Execution Summary & Environment

- **Cached Cells Validated:** 727 of 1,167 total grid cells (62.3%)
- **State Coverage:** 30 of 30 states covered (0 missing states)
- **States with < 3 Cells (2 states):**
  - `Goa`: 1 / 1
  - `Tripura`: 1 / 4
- **Feature Table Rows:** `2,918,178` rows (`727` cells × 4,014 daily observations)
- **2023 Validation Primary Set:** `232,017` cell-days, `11,732` positive fires (prevalence `5.0565%`)
- **Feature Build Time:** `62.5s` | **Peak Traced Memory:** `921.3 MB`
- **Total End-to-End Runtime:** `720.0s` (12.0 min)

---

## 2. Uncertainty: Block Bootstrap by Calendar Week of 2023 (1,000 Resamples)

1,000 block bootstrap resamples by calendar week of 2023 (seed 42) on the 2023 validation primary set. Evaluated for 5 contrasts:
- `G2 - G1` (Weather contribution over fire/spatial/calendar baseline)
- `G3 - G2` (Drought/KBDI contribution over weather features)
- `G1K - G1` (KBDI contribution over fire baseline without weather lags)
- `G2R - G2` (Weather without rainfall accumulation + KBDI vs full weather)
- `G3 - climatology` (Full feature model over historical empirical baseline)

### Bootstrap 95% Confidence Intervals

| Contrast | Point Estimate Diff | Bootstrap Mean Diff | Bootstrap Std Err | 95% Bootstrap CI | Includes 0? | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`G2 - G1`** | **+0.026046** | +0.025015 | 0.008289 | **[+0.010450, +0.041236]** | **No (False)** | Significant gain |
| **`G3 - G2`** | **+0.000670** | +0.000669 | 0.001281 | **[-0.001845, +0.003277]** | **Yes (True)** | Inconclusive (includes 0) |
| **`G1K - G1`** | **+0.010299** | +0.009718 | 0.003696 | **[+0.002054, +0.016588]** | **No (False)** | Significant gain |
| **`G2R - G2`** | **+0.000192** | +0.000141 | 0.001635 | **[-0.003101, +0.003340]** | **Yes (True)** | Inconclusive (includes 0) |
| **`G3 - climatology`** | **+0.053596** | +0.052992 | 0.006318 | **[+0.040988, +0.065615]** | **No (False)** | Significant gain |

### Findings on Uncertainty
1. **`G2 - G1` does NOT include 0:** The 95% interval `[+0.010450, +0.041236]` is strictly positive, indicating that weather features provide a measurable PR-AUC improvement over baseline fire history.
2. **`G3 - G2` includes 0:** The 95% interval `[-0.001845, +0.003277]` spans zero. The incremental contribution of KBDI over weather features remains inconclusive.
3. **`G1K - G1` does NOT include 0:** The 95% interval `[+0.002054, +0.016588]` is non-zero.
4. **`G2R - G2` includes 0:** The 95% interval `[-0.003101, +0.003340]` spans zero. Removing rainfall accumulations (precip_sum_7d/30d, days_since_rain) in favor of KBDI results in no statistically detectable difference (inconclusive).
5. **`G3 - climatology` does NOT include 0:** The 95% interval `[+0.040988, +0.065615]` is strictly positive, confirming substantial model skill over the empirical baseline.

*Traceability:* `docs/audit/partial_727_bootstrap_summary.csv` and `docs/audit/partial_727_bootstrap.csv`.

---

## 3. Ablation and Sensitivity Analysis (2023 Validation)

### Two-Stage Training (Early-Stopped on 2022) vs Fixed 150 Trees Sensitivity Check

| Group | Features | Description | Best Trees | PR-AUC (Two-Stage) | Recall@10% | Precision@5% | Brier | PR-AUC (Fixed 150) | Trees Sensitivity Δ (Fixed - Two-Stage) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `climatology` | 0 | Empirical historical cell/calendar rate | 0 | **0.224879** | 0.3764 | 0.2298 | 0.043022 | 0.224879 | +0.000000 |
| `G1` | 7 | Fire history, calendar, and static coordinates | 246 | **0.251759** | 0.4177 | 0.2505 | 0.042175 | 0.250043 | -0.001716 |
| `G1K` | 9 | G1 + KBDI features only (no weather lags) | 276 | **0.262057** | 0.4105 | 0.2529 | 0.041921 | 0.260464 | -0.001593 |
| `G2` | 19 | G1 + 12 weather features (with rain sums) | 475 | **0.277805** | 0.4238 | 0.2546 | 0.041406 | 0.276313 | -0.001492 |
| `G2R` | 18 | G2 without rainfall sums/days_since_rain + KBDI | 200 | **0.277996** | 0.4158 | 0.2534 | 0.041410 | 0.275080 | -0.002916 |
| `G3` | 21 | Full feature set (G2 + KBDI) | 376 | **0.278475** | 0.4205 | 0.2555 | 0.041395 | 0.276655 | -0.001820 |

### Tree Sensitivity Observations
- Comparing two-stage early-stopped models against a fixed 150 trees model isolates tree count noise from feature set changes.
- *Traceability:* `docs/audit/partial_727_ablation.csv` and `docs/audit/partial_727_fixed150_sensitivity.csv`.

---

## 4. Rolling-Origin Repeat (Validation Years 2021, 2022, 2023)

Evaluated across chronological rolling origins on the exact same 727 cells using primary label `new_fire_next_48h`:
- **Validation Year 2021:** Train 2015–2019, Stage 1 early-stop 2020, Stage 2 refit 2015–2020.
- **Validation Year 2022:** Train 2015–2020, Stage 1 early-stop 2021, Stage 2 refit 2015–2021.
- **Validation Year 2023:** Train 2015–2021, Stage 1 early-stop 2022, Stage 2 refit 2015–2022.
*Years 2024–2025 remain strictly locked in the test split.*

### Rolling-Origin Results Table

| Val Year | Group / Model | Features | Best Trees | PR-AUC | PR-AUC / Base | Recall@10% | Precision@5% | Brier |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2021** | `climatology` | 0 | 0 | **0.236669** | 5.183 | 0.4303 | 0.2153 | 0.038385 |
| **2021** | `G1` | 7 | 229 | **0.283038** | 6.199 | 0.4336 | 0.2343 | 0.037005 |
| **2021** | `G2` | 19 | 215 | **0.295166** | 6.464 | 0.4335 | 0.2358 | 0.036717 |
| **2021** | `G3` | 21 | 284 | **0.296254** | 6.488 | 0.4397 | 0.2394 | 0.036680 |
| **2022** | `climatology` | 0 | 0 | **0.212251** | 4.787 | 0.4031 | 0.2101 | 0.038268 |
| **2022** | `G1` | 7 | 124 | **0.247102** | 5.573 | 0.4091 | 0.2248 | 0.037182 |
| **2022** | `G2` | 19 | 292 | **0.250895** | 5.658 | 0.4208 | 0.2204 | 0.037049 |
| **2022** | `G3` | 21 | 257 | **0.253872** | 5.725 | 0.4172 | 0.2267 | 0.036972 |
| **2023** | `climatology` | 0 | 0 | **0.224879** | 4.447 | 0.3764 | 0.2298 | 0.043022 |
| **2023** | `G1` | 7 | 246 | **0.251759** | 4.979 | 0.4177 | 0.2505 | 0.042175 |
| **2023** | `G2` | 19 | 475 | **0.277805** | 5.494 | 0.4238 | 0.2546 | 0.041406 |
| **2023** | `G3` | 21 | 376 | **0.278475** | 5.507 | 0.4205 | 0.2555 | 0.041395 |

*Traceability:* `docs/audit/partial_727_rolling.csv`.

---

## 5. Fire-Season View: Evaluation on Peak Positive Months (2023 Validation)

### Performance Table on Peak Fire Months

| Season | Group / Model | Features | Rows | Positives | Base Rate | PR-AUC | PR-AUC / Base | Recall@10% | Precision@5% | Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2-3-4` | `climatology` | 0 | 48,543 | 5,010 | 10.3207% | **0.258981** | 2.509 | 0.3124 | 0.3900 | 0.084510 |
| `2-3-4` | `G1` | 7 | 48,543 | 5,010 | 10.3207% | **0.282300** | 2.735 | 0.3433 | 0.4030 | 0.082821 |
| `2-3-4` | `G2` | 19 | 48,543 | 5,010 | 10.3207% | **0.321990** | 3.120 | 0.3523 | 0.4204 | 0.080293 |
| `2-3-4` | `G3` | 21 | 48,543 | 5,010 | 10.3207% | **0.323792** | 3.137 | 0.3511 | 0.4219 | 0.080118 |
| `2-3-4-5` | `climatology` | 0 | 68,394 | 6,177 | 9.0315% | **0.239580** | 2.653 | 0.3032 | 0.3430 | 0.075493 |
| `2-3-4-5` | `G1` | 7 | 68,394 | 6,177 | 9.0315% | **0.266415** | 2.950 | 0.3525 | 0.3689 | 0.073818 |
| `2-3-4-5` | `G2` | 19 | 68,394 | 6,177 | 9.0315% | **0.301969** | 3.344 | 0.3578 | 0.3759 | 0.071870 |
| `2-3-4-5` | `G3` | 21 | 68,394 | 6,177 | 9.0315% | **0.302784** | 3.353 | 0.3520 | 0.3724 | 0.071799 |

*Traceability:* `docs/audit/partial_727_season.csv`.

---

## 6. Data Traceability Matrix

Every metric reported in this rehearsal document is traced directly to permanent saved audit artifacts:

| Audit Section | Metric / Table | Source File Path |
| :--- | :--- | :--- |
| **Section 1** | Baseline cell cohort (727 cells) | [partial_cells_727.json](file:///c:/Users/himan/Documents/Mini-Project/data/processed/partial_cells_727.json) |
| **Section 1** | Partial feature panel (2,918,178 rows) | [features_partial_727.parquet](file:///c:/Users/himan/Documents/Mini-Project/data/processed/features_partial_727.parquet) |
| **Section 2** | Bootstrap draws (1,000 resamples) | [partial_727_bootstrap.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_727_bootstrap.csv) |
| **Section 2** | 95% Confidence Intervals & zero checks | [partial_727_bootstrap_summary.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_727_bootstrap_summary.csv) |
| **Section 3** | Two-stage feature ablation | [partial_727_ablation.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_727_ablation.csv) |
| **Section 3** | Fixed-150 tree sensitivity check | [partial_727_fixed150_sensitivity.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_727_fixed150_sensitivity.csv) |
| **Section 4** | Rolling-origin 2021 & 2022 evaluations | [partial_727_rolling.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_727_rolling.csv) |
| **Section 5** | Fire-season evaluations (months 2-3-4 & 2-3-4-5) | [partial_727_season.csv](file:///c:/Users/himan/Documents/Mini-Project/docs/audit/partial_727_season.csv) |