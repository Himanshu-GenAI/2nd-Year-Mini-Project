# Objective O1 Operational Target Options

**Scope:** Formulation of operational evaluation criteria for Objective O1 (Wildfire Early Alert System) prior to test split unfreezing (2024–2025).

---

## 1. Empirical Performance Gap: Model v1 vs. Climatology

Across all evaluation years and seasonal subsets on the primary target label (`new_fire_next_48h`, primary ignition evaluation set, zero lookahead leakage, test period locked):

| Evaluation Window | Cohort Rows | Positives | Climatology PR-AUC | Model v1 PR-AUC | Absolute Gap | Relative Gain (%) | Climatology Rec@10% | Model v1 Rec@10% | Rec@10% Gap |
|---|---|---|---|---|---|---|---|---|---|
| **2023 Validation Year** | 373,605 | 18,410 | 0.219835 | 0.278369 | **+0.058534** | **+26.6%** | 0.3770 | 0.4328 | **+0.0558** |
| **2021 Rolling Origin** | 373,605 | 16,569 | 0.235573 | 0.292617 | **+0.057044** | **+24.2%** | 0.4071 | 0.4359 | **+0.0288** |
| **2022 Rolling Origin** | 373,605 | 16,688 | 0.211115 | 0.249524 | **+0.038409** | **+18.2%** | 0.4053 | 0.4245 | **+0.0192** |
| **Peak Fire Season (Feb–Apr 2023)** | 78,132 | 7,981 | 0.252160 | 0.325606 | **+0.073446** | **+29.1%** | 0.3112 | 0.3585 | **+0.0472** |

> [!IMPORTANT]
> **Observation on Recall@10%:**
> Notice that while PR-AUC reliably and substantially exceeds Climatology across every year (+18.2% to +26.6%), **Recall at top 10% has not consistently beaten Climatology by wide margins**. In rolling year 2022, the gain in Recall@10% was only +0.0192 (+1.92 percentage points, or +4.7% relative gain), and in 2021 it was +0.0288. Because climatological spatial fire frequency already concentrates in chronic fire hotspots, top-decile capacity capture is partially saturated by geography alone. Targets formulated solely on Recall@10% risk fragility or failure.

---

## 2. Three Candidate Wordings for Objective O1

### Option A: National Full-Period PR-AUC Target
> *"Model v1 achieves a PR-AUC of at least 15.0% above Climatological Baseline across the full locked test period (2024–2025) on the primary ignition evaluation set (`new_fire_next_48h`, primary set)."*
- **Empirical Basis:** Minimum rolling-year relative gain was +18.2% (2022); 2023 achieved +26.6%. A +15.0% threshold provides a 3.2% buffer against inter-annual climate anomalies.
- **Risk Assessment (2 sentences):**
  This option carries low operational deployment risk because Model v1 consistently clears +18% PR-AUC margin in all historical validation and rolling-origin test years. However, its primary risk is that national full-year evaluation pools quiet monsoon months with peak burning months, diluting seasonal early-warning salience.

### Option B: Peak Fire-Season (Feb–Apr) Focused Target
> *"During the active national fire season (February 1 to April 30 of each test year), Model v1 achieves a PR-AUC of at least 20.0% above Climatological Baseline on the primary ignition set."*
- **Empirical Basis:** Feb–Apr 2023 achieved +29.1% relative gain (+0.0734 PR-AUC margin over Climatology).
- **Risk Assessment (2 sentences):**
  This option directly targets the operational intervention window when forest departments deploy seasonal rapid-response teams and early alerts matter most. However, focusing solely on February–April risks penalizing the system if an anomalous pre-monsoon rain event abruptly truncates the seasonal fire peak in a test year.

### Option C: Recommended Balanced Hybrid Target (Recommended)
> *"Model v1 achieves both: (1) a full-period PR-AUC at least 15.0% above Climatological Baseline on the 2024–2025 test period, and (2) a Brier calibration score strictly lower than Climatological Baseline (< 0.0420) across all evaluated cell-days."*
- **Empirical Basis:** Model v1 achieved Brier = 0.040351 on 2023 (vs 0.042091 Climatology) and PR-AUC = 0.278369 (+26.6%).
- **Risk Assessment (2 sentences):**
  This wording balances discrimination power (PR-AUC margin) with operational probability reliability (Brier score), preventing uncalibrated overconfident alerting. It avoids conditioning success on brittle single-decile recall targets that vary with regional fire clustering.
