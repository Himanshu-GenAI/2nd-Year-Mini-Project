# Full-Cell Run Status Report (PS-113)

**Date:** 2026-11-21 | **Scope:** 1,167 All-India Grid Cells | **Split Status:** Test Locked (2024–2025)

The full-cell training and evaluation workflow for the India Wildfire Early Alert system is complete across all 1,167 grid cells (3,270,090 km²). All 4,383 weather days per cell were validated without missing dates or NaNs. The 21-feature panel (`data/processed/features.parquet`, 4,684,338 rows) strictly enforces temporal precedence: training is limited to 2015–2022, 2023 serves as validation, and 2024–2025 remains locked (`ALLOW_TEST_SPLIT` unset).

### Key Empirical Findings
1. **Weather Signal:** Feature group G2 (+12 weather features) significantly outperforms static/fire-history G1 (+0.0262 PR-AUC; 95% bootstrap CI [+0.0094, +0.0422]).
2. **Drought Signal:** Keetch-Byram Drought Index (G1K vs G1) demonstrates significant standalone gain (+0.0088 PR-AUC; [+0.0015, +0.0139]). When combined with full temperature and precipitation (G3 vs G2), incremental gain is -0.0001 (statistically inconclusive CI [-0.0024, +0.0021]), but G3 is retained per D18 for physical drought sensitivity.
3. **Winner Selection:** `xgb_v1_G3_tuned` (Trial 04: depth=10, lr=0.05, colsample=0.6, reg_lambda=20, 455 trees) won under D18 with PR-AUC = **0.278369**, Recall@10% = **0.4328**, and Brier = **0.040351**, outperforming Climatology (0.2198), Logistic Regression (0.2167), and Random Forest (0.2667).
4. **Calibration:** 4-quarter blocked temporal CV showed uncalibrated raw scores (Brier 0.040351) beat isotonic post-processing (0.040369); raw scores are preserved per Decision D20.

### Traceability Table

| Artifact | Location | SHA-256 / Identifier | Source Script |
|---|---|---|---|
| Final Cell List | `data/processed/cells_final.json` | `3d961ea51fb9fb3da5fd5fb8930017d85c44956fb069eb6754beea91c614a73e` | `scripts/run_cache_validation.py` |
| Full Features | `data/processed/features.parquet` | 4,684,338 rows (92.04 MB) | `src/features/build_features.py` |
| Tuning Trials | `docs/audit/tuning_trials.csv` | 20 trials, Trial 04 winner | `scripts/run_phase_c.py` |
| Ablation Results | `docs/audit/full_run_ablation.csv` | 7 model configurations | `scripts/run_phase_d.py` |
| Bootstrap CIs | `docs/audit/full_run_bootstrap_summary.csv` | 1,000 calendar-week blocks | `scripts/run_phase_d.py` |
| Rolling Origin | `docs/audit/full_run_rolling.csv` | 2021 & 2022 validation repeats | `scripts/run_phase_d.py` |
| Slice Breakdowns | `docs/audit/model_v1_breakdown.csv` | 15 temporal & geographic slices | `scripts/run_phase_e.py` |
| Model Bundle | `models/model_v1/model.json` | `7bc35c76e25cd7b3c1f325083cc07c70bc271d50995540ff70d434778068b385` | `scripts/run_phase_e.py` |
| Tier Thresholds | `configs/tiers_v1_preview.yaml` | P90=0.133950, P97=0.273928, P99.5=0.469649 | `scripts/run_phase_e.py` |
| Dev Predictions | `data/processed/predictions/model_v1_dev.parquet` | 373,605 cell-days | `scripts/run_phase_f.py` |
| Decision D20 | `docs/decisions.md` | Model v1 Winner Selection & Freezing | Team Consensus |
