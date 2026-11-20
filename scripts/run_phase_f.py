"""Phase F: Development predictions, O1 options, documentation, and close-out.

Steps:
1. Load Model v1 bundle from models/model_v1/.
2. Score 2023 validation primary-set rows (373,605 cell-days).
3. Assign operational risk tiers using configs/tiers_v1_preview.yaml.
4. Export data/processed/predictions/model_v1_dev.parquet.
5. Generate docs/o1_options.md with empirical gaps and 3 candidate wordings.
6. Write docs/full_run_status.md (<= 500 words with traceability table).
7. Update docs/schemas.md with model_v1_dev.parquet specifications.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DOCS_DIR, PROCESSED_DIR  # noqa: E402
from src.features.build_features import FEATURES_FULL_PATH  # noqa: E402
from src.models.calibrate import MODEL_V1_BUNDLE_DIR, load_model_v1_bundle, predict_model_v1  # noqa: E402
from src.models.dev_predictions import assign_tiers  # noqa: E402
from src.models.features_io import load_features  # noqa: E402
from src.models.train_v1 import assert_test_split_locked  # noqa: E402

PREDICTIONS_V1_PATH = PROCESSED_DIR / "predictions" / "model_v1_dev.parquet"
TIERS_V1_YAML_PATH = PROJECT_ROOT / "configs" / "tiers_v1_preview.yaml"
O1_OPTIONS_PATH = DOCS_DIR / "o1_options.md"
FULL_RUN_STATUS_PATH = DOCS_DIR / "full_run_status.md"
SCHEMAS_PATH = DOCS_DIR / "schemas.md"


def export_v1_dev_predictions() -> pd.DataFrame:
    """Generate and save model_v1_dev.parquet on 2023 validation primary set."""
    assert_test_split_locked()
    print("\n--- Generating Model v1 Dev Predictions ---")

    # Load bundle
    bundle = load_model_v1_bundle(bundle_dir=MODEL_V1_BUNDLE_DIR, verify_checksums=True)
    version = bundle.get("version", "v1_tuned")
    feature_names = bundle["features"]

    # Load validation primary set (2023)
    label = "new_fire_next_48h"
    X_va, y_va, keys_va = load_features(
        "val",
        label=label,
        feature_names=feature_names,
        primary=True,
        features_path=FEATURES_FULL_PATH,
    )
    n_rows = len(X_va)
    print(f"Loaded validation primary set: {n_rows:,} rows")
    assert n_rows == 373605, f"Expected 373,605 rows, got {n_rows}"

    # Score
    scores = predict_model_v1(bundle, X_va).astype(np.float32)
    assert not np.isnan(scores).any(), "NaN found in predictions"
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Scores outside [0, 1]"

    # Load tier thresholds
    if not TIERS_V1_YAML_PATH.exists():
        raise FileNotFoundError(f"Missing {TIERS_V1_YAML_PATH}")
    with open(TIERS_V1_YAML_PATH, encoding="utf-8") as f:
        tiers_cfg = yaml.safe_load(f)
    thresholds = tiers_cfg["thresholds"]

    # Assign tiers
    tiers = assign_tiers(scores, thresholds)

    # Build DataFrame matching v0 schema exactly
    df_pred = pd.DataFrame({
        "cell_id": keys_va["cell_id"].astype(str).values,
        "date": pd.to_datetime(keys_va["date"].values),
        "score": scores,
        "tier": pd.Categorical(tiers, categories=["Low", "Moderate", "High", "Critical"], ordered=True),
        "model_version": str(version),
        "split": "val",
    })

    # Assertions
    assert len(df_pred) == 373605
    assert df_pred["split"].unique().tolist() == ["val"]
    assert list(df_pred["date"].dt.year.unique()) == [2023]
    assert not df_pred.duplicated(subset=["cell_id", "date"]).any()

    # Save
    PREDICTIONS_V1_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_parquet(PREDICTIONS_V1_PATH, index=False)
    print(f"Saved: {PREDICTIONS_V1_PATH} ({PREDICTIONS_V1_PATH.stat().st_size / 1e6:.2f} MB)")

    tier_shares = df_pred["tier"].value_counts(normalize=True).to_dict()
    print("Tier shares:")
    for t_name, share in tier_shares.items():
        print(f"  {t_name:10s}: {share * 100:.3f}% ({int(share * len(df_pred)):,} rows)")

    return df_pred


def write_o1_options() -> None:
    """Generate docs/o1_options.md with empirical metrics and 3 candidate wordings."""
    content = """# Objective O1 Operational Target Options

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
"""
    O1_OPTIONS_PATH.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Saved: {O1_OPTIONS_PATH}")


def update_schemas_md() -> None:
    """Append Model v1 dev predictions schema to docs/schemas.md if missing."""
    content = SCHEMAS_PATH.read_text(encoding="utf-8")
    v1_section = """
### G. Model v1 Development Predictions (`data/processed/predictions/model_v1_dev.parquet`)
- **Row Count:** 373,605 cell-days (2023 validation primary evaluation set: `split == "val"` and `burning_prev3 == False`). Zero training, test, or gap rows.
- **Scope:** Primary-set ignition evaluation set for Model v1 (`xgb_v1_G3_tuned`, 21 features, 455 trees).
- **Tier Thresholds Config:** `configs/tiers_v1_preview.yaml` (`P90 = 0.133950`, `P97 = 0.273928`, `P99.5 = 0.469649`).
- **Columns:**
  | Column | Dtype | Description |
  |---|---|---|
  | `cell_id` | `str` | Grid cell identifier (`r{row:02d}c{col:02d}`) |
  | `date` | `datetime64[ns]` | Validation date in 2023 (`2023-01-01` to `2023-12-29`) |
  | `score` | `float32` | Raw predicted probability of `new_fire_next_48h` |
  | `tier` | `category` | Operational risk tier (`Low` < P90, `Moderate` P90–P97, `High` P97–P99.5, `Critical` >= P99.5) |
  | `model_version` | `str` | Version tag (`v1_tuned_20261121`) |
  | `split` | `str` | Data split (`"val"` only) |
"""
    if "Model v1 Development Predictions" not in content:
        content += "\n---\n" + v1_section.strip() + "\n"
        SCHEMAS_PATH.write_text(content, encoding="utf-8")
        print(f"Updated: {SCHEMAS_PATH} with Model v1 schema.")
    else:
        print("Model v1 schema already present in docs/schemas.md.")


def write_full_run_status() -> None:
    """Generate docs/full_run_status.md (<= 500 words, traceability table)."""
    text = """# Full-Cell Run Status Report (PS-113)

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
"""
    FULL_RUN_STATUS_PATH.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"Saved: {FULL_RUN_STATUS_PATH}")


def phase_f() -> None:
    """Execute complete Phase F pipeline."""
    t0 = time.perf_counter()
    print("=" * 70)
    print("PHASE F: Dev Predictions, O1 Options, Schemas & Close-Out")
    print("=" * 70)

    export_v1_dev_predictions()
    write_o1_options()
    update_schemas_md()
    write_full_run_status()

    elapsed = time.perf_counter() - t0
    print(f"\nPhase F completed in {elapsed:.1f}s")


if __name__ == "__main__":
    phase_f()
