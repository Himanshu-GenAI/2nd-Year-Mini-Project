# Operational Confusion Matrices - Model v1 (`xgb_v1_G3_tuned`)

**Evaluation Set:** 2023 Validation Primary Ignition Set (`new_fire_next_48h`, `split == 'val'`, `burning_prev3 == False`)  
**Scope:** 1,167 All-India Grid Cells (373,605 total cell-days; 18,410 actual new fire events, **4.93% base rate**)  
**Model:** `xgb_v1_G3_tuned` (21 features, 455 trees, saved in `models/model_v1/`)  
**Test Split Status:** 2024–2025 strictly locked (`ALLOW_TEST_SPLIT` unset).

---

## 1. Methodological & Evaluation Caveats

1. **Extreme Class Imbalance:**
   - Only 4.93% of cell-days have active new ignitions (18,410 positives vs. 355,195 negatives).
   - **Accuracy is removed as a headline metric:** A trivial "predict all zeros / no fire" model achieves **95.07% accuracy** while providing zero operational utility. Headlines instead focus on **Precision (PPV)**, **Recall (Sensitivity)**, **False Alarm Ratio (FAR)**, and **Precision Lift over Base Rate**.
2. **Pooled vs. Per-Day Recall at Top 10%:**
   - **Pooled Recall at Top 10% (`50.55%`):** Evaluated against a single fixed global threshold ($P_{90} = 0.1340$). Flags any cell-day nationwide with score $\ge 0.1340$. Because fire activity clusters in peak pre-monsoon months, more than 10% of cells are flagged on high-fire days and fewer than 10% on quiet monsoon days, capturing 9,307 / 18,410 (50.55%) of all annual ignitions.
   - **Per-Day Top 10% Mean Recall (`43.28%`):** Evaluated against a fixed daily operational budget (e.g., forest surveillance teams inspecting the top 10% highest-risk cells on each calendar day). Dates with zero fires are excluded from the mean. This is the primary metric reported in benchmark leaderboards (`docs/results.csv`, `docs/model_v1_report.md`).

---

## 2. Multi-Tier Cross Tabulation (4 Operational Tiers)

Thresholds defined in `configs/tiers_v1_preview.yaml` from 2023 primary-set score percentiles ($P_{90} = 0.133950$, $P_{97} = 0.273928$, $P_{99.5} = 0.469649$):

| Predicted Tier | Threshold Range | Actual: No Fire (0) | Actual: New Fire (1) | Total Cell-Days | Precision (Hit Rate) | Share of Annual Fires |
|---|---|---|---|---|---|---|
| **Low** | Score $< 0.1340$ | 327,141 | 9,103 | 336,244 (90.00%) | 2.71% | 49.45% |
| **Moderate** | $0.1340 \le \text{Score} < 0.2739$ | 21,120 | 5,032 | 26,152 (7.00%) | 19.24% | 27.33% |
| **High** | $0.2739 \le \text{Score} < 0.4696$ | 6,068 | 3,272 | 9,340 (2.50%) | 35.03% | 17.77% |
| **Critical** | $\text{Score} \ge 0.4696$ | 866 | 1,003 | 1,869 (0.50%) | **53.67%** | 5.45% |
| **Total** | | **355,195** | **18,410** | **373,605 (100%)** | **4.93%** | **100.00%** |

---

## 3. Operational Binary Alert Confusion Matrices

### A. Surveillance Alert Threshold: Top 10% Flagged (Moderate + High + Critical, Score $\ge 0.1340$)
*Operational context: General alert level for ground patrols and enhanced monitoring.*

| | Predicted Negative (Score $< 0.1340$) | Predicted Alert (Score $\ge 0.1340$) | Total |
|---|---|---|---|
| **Actual: No Fire (0)** | 327,141 (TN) | 28,054 (FP) | 355,195 |
| **Actual: New Fire (1)** | 9,103 (FN) | 9,307 (TP) | 18,410 |
| **Total** | 336,244 | 37,361 | 373,605 |

- **True Positives (TP):** 9,307
- **False Positives (FP):** 28,054
- **True Negatives (TN):** 327,141
- **False Negatives (FN):** 9,103
- **Pooled Recall:** **50.55%** (9,307 / 18,410)
- **Per-Day Top 10% Mean Recall:** **43.28%**
- **Precision (PPV):** **24.91%** (5.05× lift over 4.93% base rate)
- **Specificity (TNR):** **92.10%** (327,141 / 355,195)
- **False Alarm Ratio (FAR):** **75.09%** (28,054 / 37,361)
- **F1 Score:** 0.3338

---

### B. Response Alert Threshold: Top 3% Flagged (High + Critical, Score $\ge 0.2739$)
*Operational context: Rapid deployment of fire suppression crews and standby tankers.*

| | Predicted Negative (Score $< 0.2739$) | Predicted Alert (Score $\ge 0.2739$) | Total |
|---|---|---|---|
| **Actual: No Fire (0)** | 348,261 (TN) | 6,934 (FP) | 355,195 |
| **Actual: New Fire (1)** | 14,135 (FN) | 4,275 (TP) | 18,410 |
| **Total** | 362,396 | 11,209 | 373,605 |

- **True Positives (TP):** 4,275
- **False Positives (FP):** 6,934
- **True Negatives (TN):** 348,261
- **False Negatives (FN):** 14,135
- **Pooled Recall:** **23.22%** (4,275 / 18,410)
- **Precision (PPV):** **38.14%** (7.74× lift over base rate)
- **Specificity (TNR):** **98.05%**
- **False Alarm Ratio (FAR):** **61.86%**
- **F1 Score:** 0.2887

---

### C. Emergency Alert Threshold: Top 0.5% Flagged (Critical Tier, Score $\ge 0.4696$)
*Operational context: Extreme danger alerts, forest closure, pre-emptive water drops.*

| | Predicted Negative (Score $< 0.4696$) | Predicted Alert (Score $\ge 0.4696$) | Total |
|---|---|---|---|
| **Actual: No Fire (0)** | 354,329 (TN) | 866 (FP) | 355,195 |
| **Actual: New Fire (1)** | 17,407 (FN) | 1,003 (TP) | 18,410 |
| **Total** | 371,736 | 1,869 | 373,605 |

- **True Positives (TP):** 1,003
- **False Positives (FP):** 866
- **True Negatives (TN):** 354,329
- **False Negatives (FN):** 17,407
- **Pooled Recall:** **5.45%** (1,003 / 18,410)
- **Precision (PPV):** **53.67%** (10.89× lift over base rate; majority of alerts are verified fires)
- **Specificity (TNR):** **99.76%**
- **False Alarm Ratio (FAR):** **46.33%**
- **F1 Score:** 0.0989

---

### D. Benchmark Reference: Standard 0.50 Threshold (Score $\ge 0.50$)
*Operational context: Standard uncalibrated classification threshold for academic comparison.*

| | Predicted Negative (Score $< 0.50$) | Predicted Alert (Score $\ge 0.50$) | Total |
|---|---|---|---|
| **Actual: No Fire (0)** | 354,616 (TN) | 579 (FP) | 355,195 |
| **Actual: New Fire (1)** | 17,653 (FN) | 757 (TP) | 18,410 |
| **Total** | 372,269 | 1,336 | 373,605 |

- **True Positives (TP):** 757
- **False Positives (FP):** 579
- **True Negatives (TN):** 354,616
- **False Negatives (FN):** 17,653
- **Pooled Recall:** 4.11%
- **Precision (PPV):** 56.66%
- **Specificity (TNR):** 99.84%
- **False Alarm Ratio (FAR):** 43.34%
- **F1 Score:** 0.0767
