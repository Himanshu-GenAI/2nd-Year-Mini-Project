"""Phase E: Model v1 Calibration, Tiers Preview, Saved Model Bundle, and Report.

Steps:
1. Load winner hyperparameters (Trial 04 on G3).
2. Fit final Model v1 on 2015-2022 primary training set (455 trees).
3. Score 2023 validation primary set.
4. Run 4-quarter blocked CV for isotonic probability calibration.
5. Compute tier thresholds (P90, P97, P99.5) and save configs/tiers_v1_preview.yaml.
6. Compute slice breakdowns and save docs/audit/model_v1_breakdown.csv.
7. Save Model v1 bundle to models/model_v1/ with SHA-256 checksums.
8. Verify bundle reload and checksum integrity.
9. Update docs/decisions.md with Decision D20.
10. Generate docs/model_v1_report.md.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from scripts.make_model_v1_report import build_model_v1_report  # noqa: E402
from src.config import DOCS_AUDIT_DIR, DOCS_DIR  # noqa: E402
from src.data.grid import load_grid  # noqa: E402
from src.features.build_features import ALL_FEATURE_NAMES, FEATURES_FULL_PATH  # noqa: E402
from src.models.calibrate import (  # noqa: E402
    BREAKDOWN_CSV_PATH,
    MODEL_V1_BUNDLE_DIR,
    RELIABILITY_PNG_PATH,
    TIERS_PREVIEW_V1_PATH,
    blocked_quarterly_cv,
    compute_tiers_and_breakdowns,
    load_model_v1_bundle,
    predict_model_v1,
    save_model_v1_bundle,
)
from src.models.evaluate import brier, pr_auc, recall_at_top_frac  # noqa: E402
from src.models.features_io import load_features  # noqa: E402
from src.models.train_v1 import assert_test_split_locked  # noqa: E402

TIERS_V1_PREVIEW_PATH = PROJECT_ROOT / "configs" / "tiers_v1_preview.yaml"
DECISIONS_MD_PATH = DOCS_DIR / "decisions.md"


def phase_e() -> dict[str, Any]:
    """Execute complete Phase E pipeline."""
    t0 = time.perf_counter()
    assert_test_split_locked()

    print("\n" + "=" * 70)
    print("PHASE E: Model v1 Calibration, Tiers, Saved Bundle & Report")
    print("=" * 70)

    # 1. Load winner hyperparameters
    trials_csv = DOCS_AUDIT_DIR / "tuning_trials.csv"
    if not trials_csv.exists():
        raise FileNotFoundError(f"Missing {trials_csv}")
    trials_df = pd.read_csv(trials_csv)
    best_row = trials_df.iloc[0]
    best_trees = int(best_row["best_trees"])
    cfg = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "min_child_weight": float(best_row["min_child_weight"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "reg_lambda": float(best_row["reg_lambda"]),
    }
    print(f"Winner: Trial {int(best_row['trial'])} on G3 ({len(ALL_FEATURE_NAMES)} features)")
    print(f"Hyperparameters: {cfg}, trees: {best_trees}")

    # 2. Fit final Model v1 booster on full 2015-2022 primary train set
    print("\nLoading 2015-2022 train and 2023 val primary splits...")
    label = "new_fire_next_48h"
    X_tr, y_tr, keys_tr = load_features(
        "train", label=label, feature_names=ALL_FEATURE_NAMES,
        primary=True, features_path=FEATURES_FULL_PATH,
    )
    X_va, y_va, keys_va = load_features(
        "val", label=label, feature_names=ALL_FEATURE_NAMES,
        primary=True, features_path=FEATURES_FULL_PATH,
    )
    print(f"  Train: {len(X_tr):,} rows ({int(y_tr.sum()):,} pos)")
    print(f"  Val:   {len(X_va):,} rows ({int(y_va.sum()):,} pos)")

    print(f"\nFitting final XGBoost model ({best_trees} trees on {len(X_tr):,} rows)...")
    t_fit = time.perf_counter()
    model = xgb.XGBClassifier(
        n_estimators=best_trees,
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        min_child_weight=cfg["min_child_weight"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_lambda=cfg["reg_lambda"],
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, verbose=False)
    dur_fit = time.perf_counter() - t_fit
    print(f"Model fit complete in {dur_fit:.1f}s")

    # 3. Score 2023 validation primary set
    print("\nScoring 2023 validation primary set...")
    raw_scores = model.predict_proba(X_va)[:, 1]
    df_val = keys_va.copy()
    df_val[label] = y_va.values
    df_val["score"] = raw_scores
    raw_prauc = float(pr_auc(df_val, label, "score"))
    raw_brier_score = float(brier(df_val, label, "score"))
    raw_rec10, _ = recall_at_top_frac(df_val, label, "score", frac=0.10)
    print(f"  Raw 2023 Val: PR-AUC={raw_prauc:.6f}, Recall@10%={raw_rec10:.4f}, Brier={raw_brier_score:.6f}")

    # 4. Run 4-quarter blocked CV for isotonic probability calibration
    print("\nRunning 4-quarter temporal blocked CV for isotonic calibration...")
    cv_res = blocked_quarterly_cv(
        df_val=df_val,
        label=label,
        score_col="score",
        date_col="date",
    )
    print(f"  Blocked CV: raw Brier = {cv_res['raw_brier_cv']:.6f}, cal Brier = {cv_res['cal_brier_cv']:.6f}")
    print(f"  Use calibrator? {cv_res['use_calibrator']}")
    print("\nQuarterly Folds:")
    print(cv_res["folds_df"].to_string(index=False))

    calibrator = cv_res["calibrator"]
    cal_scores = cv_res["oof_scores"] if cv_res["use_calibrator"] else raw_scores
    df_val["score_cal"] = cal_scores
    eval_score_col = "score_cal" if cv_res["use_calibrator"] else "score"

    # 5. Compute tier thresholds (P90, P97, P99.5) and slice breakdowns
    print("\nComputing operational tier thresholds and slice breakdowns...")
    grid_df = load_grid()
    version_tag = f"v1_tuned_{date.today().strftime('%Y%m%d')}"

    thresholds, breakdown_df = compute_tiers_and_breakdowns(
        df_val_primary=df_val,
        score_col=eval_score_col,
        label=label,
        grid_df=grid_df,
        model_version=version_tag,
        tiers_path=TIERS_V1_PREVIEW_PATH,
        breakdown_path=BREAKDOWN_CSV_PATH,
        reliability_path=RELIABILITY_PNG_PATH,
    )
    # Also save to TIERS_PREVIEW_V1_PATH (configs/tiers_preview.yaml)
    shutil.copyfile(TIERS_V1_PREVIEW_PATH, TIERS_PREVIEW_V1_PATH)
    print(f"  Tier Cutoffs: {thresholds}")
    print(f"  Saved tiers preview to {TIERS_V1_PREVIEW_PATH} and {TIERS_PREVIEW_V1_PATH}")
    print(f"  Saved slice breakdown ({len(breakdown_df)} slices) to {BREAKDOWN_CSV_PATH}")

    # 6. Save Model v1 bundle with SHA-256 checksums
    print(f"\nPackaging Model v1 bundle in {MODEL_V1_BUNDLE_DIR}...")
    fit_dates = {
        "train_start": "2015-01-03",
        "train_end": "2022-12-29",
        "val_start": "2023-01-01",
        "val_end": "2023-12-29",
        "n_train_rows": len(X_tr),
        "n_val_rows": len(X_va),
        "n_cells": 1167,
    }
    checksums = save_model_v1_bundle(
        model=model,
        feature_names=ALL_FEATURE_NAMES,
        fit_dates=fit_dates,
        params={**cfg, "n_estimators": best_trees},
        version=version_tag,
        calibrator=calibrator,
        bundle_dir=MODEL_V1_BUNDLE_DIR,
    )
    print("Saved bundle files and SHA-256 checksums:")
    for fname, sha in sorted(checksums.items()):
        print(f"  {fname:20s}: {sha}")

    # 7. Verification test: reload bundle and check bitwise/numerical inference match
    print("\nVerifying model bundle reload and inference reproducibility...")
    bundle = load_model_v1_bundle(bundle_dir=MODEL_V1_BUNDLE_DIR, verify_checksums=True)
    reloaded_scores = predict_model_v1(bundle, X_va.iloc[:1000])
    expected_sample = df_val[eval_score_col].values[:1000]
    np.testing.assert_allclose(reloaded_scores, expected_sample, rtol=1e-5, atol=1e-5)
    print("  PASS: Reloaded bundle predictions match exactly (sample of 1,000 rows).")

    # 8. Record Decision D20 in docs/decisions.md
    print(f"\nRecording Decision D20 in {DECISIONS_MD_PATH}...")
    d20_text = (
        f"| **D20** | {date.today()} | Model v1 Winner Selection and Freezing: `xgb_v1_G3_tuned` "
        f"(21 features, {best_trees} trees, max_depth={cfg['max_depth']}, lr={cfg['learning_rate']}, "
        f"colsample={cfg['colsample_bytree']}, lambda={cfg['reg_lambda']}). "
        f"SHA-256 (`model.json`): `{checksums.get('model.json')}`. | "
        f"Ranked 1st in Phase D full-cell ablation (PR-AUC: {raw_prauc:.6f}, Recall@10%: {raw_rec10:.4f}, "
        f"+0.0585 over Climatology). Significant gains across bootstrap contrasts; all 1,167 cells validated. | "
        f"`docs/audit/full_run_diagnostics.md`, `models/model_v1/checksums.json` | Team | `frozen` |\n"
    )

    dec_content = DECISIONS_MD_PATH.read_text(encoding="utf-8")
    if "D20" not in dec_content:
        # Insert D20 row after D19 row
        marker = "| **D19** |"
        idx = dec_content.find(marker)
        if idx != -1:
            end_of_line = dec_content.find("\n", idx)
            dec_content = dec_content[:end_of_line + 1] + d20_text + dec_content[end_of_line + 1:]
            DECISIONS_MD_PATH.write_text(dec_content, encoding="utf-8")
            print("  Appended D20 to docs/decisions.md.")
    else:
        print("  D20 already present in docs/decisions.md.")

    # 9. Build Model v1 report
    print("\nGenerating Model v1 report...")
    report_path = DOCS_DIR / "model_v1_report.md"
    report_text = build_model_v1_report(
        results_csv=DOCS_DIR / "results.csv",
        ablation_csv=DOCS_AUDIT_DIR / "full_run_ablation.csv",
        breakdown_csv=BREAKDOWN_CSV_PATH,
        importance_csv=DOCS_AUDIT_DIR / "full_run_feature_importance.csv",
        tiers_yaml=TIERS_V1_PREVIEW_PATH,
        output_md=report_path,
        partial_note=None,
    )
    print(f"Saved {report_path} ({len(report_text)} chars)")

    elapsed = time.perf_counter() - t0
    result = {
        "status": "PASS",
        "model_version": version_tag,
        "n_trees": best_trees,
        "raw_pr_auc": round(raw_prauc, 6),
        "raw_recall_top10": round(raw_rec10, 4),
        "calibrated": cv_res["use_calibrator"],
        "checksums": checksums,
        "elapsed_s": round(elapsed, 1),
    }
    print(f"\nPhase E completed in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    result = phase_e()
    print(f"\n=== Phase E result: {json.dumps(result, indent=2)} ===")
