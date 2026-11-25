# Test Protocol for Model v1 One-Shot Evaluation (PS-113)

**Document Status:** Pre-Registered Protocol | **Target Model:** Model v1 (`v1_tuned_20261121`, G3 tuned XGBoost) | **Date:** 2026-11-26

This protocol specifies the locked evaluation procedure for testing Model v1 on the out-of-sample test period (`2024-01-01` to `2025-12-29`). The evaluation must be executed strictly once using `scripts/run_test_once.py`.

---

## 1. Target Labels, Splits, and Cohorts

1. **Primary Label:** `new_fire_next_48h`
   - Evaluated on the **Primary Set**: rows where `burning_prev3 == False` (new ignitions in cells not actively burning in the preceding 3 days).
2. **Secondary Label:** `fire_next_48h`
   - Evaluated on **All Rows** (both newly ignited and continuing fire events).
3. **Test Period:**
   - `2024-01-01` to `2025-12-29` (729 dates x 1,167 cells), scored strictly once.
4. **Baseline Model:**
   - **Climatology Baseline**, fitted strictly on historical observations from `2015-01-03` to `2022-12-29` (no test period data leakage).

---

## 2. Evaluation Metrics & Stratification

The following metrics are computed for:
- **Pooled Test Period** (`2024–2025`)
- **Calendar Year 2024** separately
- **Calendar Year 2025** separately

### Required Metrics
1. **PR-AUC** (Area Under the Precision-Recall Curve)
2. **PR-AUC over Base Rate** ($\text{PR-AUC} / \text{Base Rate}$)
3. **Per-Day Recall at Top 10%** (mean daily recall across active cells prioritized in the top decile of risk)
4. **Precision at Top 5%** (pooled positive predictive value in the top 5% highest-risk cell-days)
5. **Brier Score** (mean squared probability error)
6. **Pooled Confusion Matrices** at the three frozen tier thresholds (P90 = `0.133950`, P97 = `0.273928`, P99.5 = `0.469649`)
7. **Block Bootstrap Uncertainty:** 1,000 calendar-week block resamples (seed 42) computing 95% confidence intervals for:
   - Model PR-AUC minus Climatology PR-AUC
   - Model per-day Recall@10% minus Climatology per-day Recall@10%

---

## 3. Pre-Registered Success Rule (Objective O1)

The success of Model v1 on the India Wildfire Early Alert operational challenge is governed strictly by the following pre-registered success rule:

> *"On the 2024-2025 test period, new-fire label, primary set: PR-AUC at least 15% above climatology, with the 95% week-block bootstrap interval for the difference excluding 0, and per-day recall at the top 10% not below climatology."*

### Explicit Mathematical Evaluation (Unrounded Values)
To achieve a final verdict of **PASS**, all three conditions must be satisfied simultaneously using unrounded floating-point metrics:

1. **Condition (a) — PR-AUC Relative Margin:**
   $$\frac{\text{PR-AUC}_{\text{model}}}{\text{PR-AUC}_{\text{climatology}}} - 1 \ge 0.15$$
2. **Condition (b) — Statistical Significance:**
   $$\text{CI}_{95\%}^{\text{lower}}(\text{PR-AUC}_{\text{model}} - \text{PR-AUC}_{\text{climatology}}) > 0$$
3. **Condition (c) — Operational Capture Baseline:**
   $$\text{Recall@10\%}_{\text{model}} \ge \text{Recall@10\%}_{\text{climatology}}$$

---

## 4. Execution Governance & Anti-Tampering Rules

- **One-Shot Execution:** The test period is run strictly once. No model parameters, feature definitions, tier thresholds, hyperparameter settings, or evaluation rules may be modified after the test run begins.
- **Audit Lock:** Upon completion, `scripts/run_test_once.py` writes a permanent lock file `docs/audit/test_run_lock.json` capturing the execution timestamp, git commit, and artifact hashes. The script refuses to run again if this lock exists.
- **Bug Discovery Disclosure:** If a software bug is discovered post-run, both the original "before" and corrected "after" results must be fully documented side-by-side in all public reports. The pre-registered success rule cannot be altered retrospectively.
- **Environment Isolation:** The test split access environment variable (`ALLOW_TEST_SPLIT=1`) must be asserted only inside the execution process of `scripts/run_test_once.py` and must never remain set in persistent shell or CI environments.

---

## 5. Known Operational Limitations

All subsequent benchmark reports and publications must transparently restate the following operational limitations:
1. **Observed Weather vs. Numerical Weather Forecasts:** Predictions use observed ERA5 reanalysis weather through day $t$, not operational multi-day NWP meteorological forecasts.
2. **FIRMS Satellite Detection Thresholds:** NASA VIIRS 375m thermal anomalies miss under-canopy smoldering and fires obscured by persistent cloud cover or heavy smoke plumes.
3. **Agricultural Crop Stubble Confounding:** A substantial fraction of active thermal detections across northern agricultural belts (e.g. Punjab, Haryana, Indo-Gangetic plain) represents open residue burning rather than wild forest fires.
4. **Hyperparameter Tuning Context:** Model hyperparameters were tuned on the 2023 validation split, so 2023 metrics represent optimistic estimates compared to the held-out test split.
