"""Unit tests for Model v1 calibration, blocked CV, bundle integrity, and report guard (Day 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import yaml

from scripts.make_model_v1_report import build_model_v1_report
from src.features.build_features import ALL_FEATURE_NAMES
from src.models.calibrate import (
    blocked_quarterly_cv,
    compute_tiers_and_breakdowns,
    fit_isotonic,
    load_model_v1_bundle,
    predict_isotonic,
    predict_model_v1,
    save_model_v1_bundle,
)
from src.models.train_v0 import prior_correct


def test_isotonic_monotonicity_and_bounds():
    rng = np.random.default_rng(42)
    scores = rng.uniform(0.0, 1.0, size=500)
    probs_true = 1.0 / (1.0 + np.exp(-4.0 * (scores - 0.5)))
    y = (rng.random(500) < probs_true).astype(np.int64)

    iso = fit_isotonic(scores, y)
    grid = np.linspace(-0.2, 1.2, 100)
    calibrated = predict_isotonic(iso, grid)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert np.all(np.diff(calibrated) >= -1e-12)


def test_prior_correct_beta_one_round_trip():
    probs = np.array([0.0, 0.05, 0.25, 0.5, 0.9, 1.0])
    out = prior_correct(probs, beta=1.0)
    np.testing.assert_allclose(out, probs, rtol=0, atol=0)
    assert prior_correct(0.37, beta=1.0) == 0.37


def test_blocked_quarterly_cv_and_small_quarter_flag():
    rng = np.random.default_rng(42)
    # Build 2023 dates across all 4 quarters; make Q2 have <100 positives
    dates = pd.date_range("2023-01-01", "2023-12-29", freq="D")
    rows = []
    for c in [f"r00c{i:02d}" for i in range(4)]:
        for d in dates:
            # Make scores poorly calibrated (overconfident) so isotonic improves Brier in CV
            p_true = 0.35 if d.month in (3, 4, 5) else 0.03
            y_val = 1 if (d.quarter != 2 and rng.random() < p_true * 3.0) else (1 if d.quarter == 2 and rng.random() < 0.01 else 0)
            score = np.clip(0.85 if y_val == 1 else 0.25, 0.0, 1.0)
            rows.append({"cell_id": c, "date": d, "score": score, "new_fire_next_48h": y_val})
    df_val = pd.DataFrame(rows)

    res = blocked_quarterly_cv(df_val, label="new_fire_next_48h", score_col="score", min_pos_for_judge=100)
    folds = res["folds_df"]
    assert len(folds) == 4
    assert set(folds["quarter"]) == {"Q1", "Q2", "Q3", "Q4"}
    q2_note = folds.loc[folds["quarter"] == "Q2", "note"].iloc[0]
    assert q2_note == "too few to judge"
    assert res["use_calibrator"] is True
    assert res["cal_brier_cv"] < res["raw_brier_cv"]
    assert res["calibrator"] is not None

    # Now make scores already well-calibrated and small variation so calibration doesn't strictly improve or verify rule
    df_no_gain = df_val.copy()
    df_no_gain["score"] = df_no_gain.groupby(df_no_gain["date"].dt.quarter)["new_fire_next_48h"].transform("mean")
    res_no_gain = blocked_quarterly_cv(df_no_gain, label="new_fire_next_48h", score_col="score")
    assert res_no_gain["use_calibrator"] == (res_no_gain["cal_brier_cv"] < res_no_gain["raw_brier_cv"])
    if not res_no_gain["use_calibrator"]:
        assert res_no_gain["calibrator"] is None


def test_save_and_load_model_v1_bundle_and_tiers(tmp_path: Path):
    rng = np.random.default_rng(42)
    n_tr = 300
    n_va = 120
    X_tr = pd.DataFrame(rng.normal(size=(n_tr, len(ALL_FEATURE_NAMES))).astype(np.float32), columns=ALL_FEATURE_NAMES)
    y_tr = (X_tr["kbdi"].values + X_tr["fire_count_7d"].values > 0).astype(np.int64)
    X_va = pd.DataFrame(rng.normal(size=(n_va, len(ALL_FEATURE_NAMES))).astype(np.float32), columns=ALL_FEATURE_NAMES)
    y_va = (X_va["kbdi"].values + X_va["fire_count_7d"].values > 0).astype(np.int64)

    clf = xgb.XGBClassifier(n_estimators=15, max_depth=3, random_state=42, tree_method="hist", eval_metric="aucpr")
    clf.fit(X_tr, y_tr, verbose=False)
    raw_va = clf.predict_proba(X_va)[:, 1]
    iso = fit_isotonic(raw_va, y_va)
    expected_preds = predict_isotonic(iso, raw_va)

    bundle_dir = tmp_path / "model_v1"
    checksums = save_model_v1_bundle(
        model=clf,
        feature_names=ALL_FEATURE_NAMES,
        fit_dates={"train_start": "2015-01-03", "train_end": "2022-12-29", "cal_year": 2023},
        params={"max_depth": 3, "n_estimators": 15, "prior_beta": 1.0},
        version="v1_test_20261111",
        calibrator=iso,
        bundle_dir=bundle_dir,
    )
    assert (bundle_dir / "checksums.json").exists()
    assert "model.json" in checksums
    assert "calibrator.joblib" in checksums

    loaded = load_model_v1_bundle(bundle_dir, verify_checksums=True)
    # Shuffle X_va columns to prove feature order alignment in predict_model_v1
    shuffled_cols = list(reversed(ALL_FEATURE_NAMES))
    actual_preds = predict_model_v1(loaded, X_va[shuffled_cols])
    np.testing.assert_allclose(actual_preds, expected_preds, rtol=1e-6, atol=1e-6)

    # Corrupt version.txt and confirm checksum verification fails loudly
    (bundle_dir / "version.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_model_v1_bundle(bundle_dir, verify_checksums=True)

    # Test compute_tiers_and_breakdowns outputs
    val_df = pd.DataFrame(
        {
            "cell_id": [f"r00c{i % 4:02d}" for i in range(n_va)],
            "date": pd.date_range("2023-01-01", periods=n_va, freq="3D"),
            "new_fire_next_48h": y_va,
            "burning_prev3": False,
            "score": expected_preds,
        }
    )
    grid_df = pd.DataFrame({"cell_id": [f"r00c{i:02d}" for i in range(4)], "state": ["MP", "MP", "MH", "OR"]})
    tiers_yaml = tmp_path / "tiers_preview.yaml"
    brk_csv = tmp_path / "breakdown.csv"
    rel_png = tmp_path / "reliability.png"
    thresholds, brk_df = compute_tiers_and_breakdowns(
        df_val_primary=val_df,
        score_col="score",
        grid_df=grid_df,
        model_version="v1_test",
        tiers_path=tiers_yaml,
        breakdown_path=brk_csv,
        reliability_path=rel_png,
    )
    assert tiers_yaml.exists() and brk_csv.exists() and rel_png.exists()
    assert set(thresholds.keys()) == {"p90", "p97", "p99_5"}
    assert "season" in set(brk_df["dimension"])
    assert "state" in set(brk_df["dimension"])


def test_make_model_v1_report_fails_when_sources_missing(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    ablation_csv = tmp_path / "ablation.csv"
    breakdown_csv = tmp_path / "breakdown.csv"
    importance_csv = tmp_path / "feature_importance.csv"
    tiers_yaml = tmp_path / "tiers_preview.yaml"
    out_md = tmp_path / "model_v1_report.md"

    # Write only 4 of 5 required files -> must raise FileNotFoundError
    pd.DataFrame([{"model": "xgb_v1", "set_name": "primary", "pr_auc": 0.3}]).to_csv(results_csv, index=False)
    pd.DataFrame([{"group": "G3", "pr_auc": 0.3}]).to_csv(ablation_csv, index=False)
    pd.DataFrame([{"dimension": "season", "bucket": "Pre-monsoon", "pr_auc": 0.3}]).to_csv(breakdown_csv, index=False)
    pd.DataFrame([{"feature": "kbdi", "perm_importance_mean": 0.05}]).to_csv(importance_csv, index=False)

    with pytest.raises(FileNotFoundError, match="tiers_preview.yaml"):
        build_model_v1_report(
            results_csv=results_csv,
            ablation_csv=ablation_csv,
            breakdown_csv=breakdown_csv,
            importance_csv=importance_csv,
            tiers_yaml=tiers_yaml,
            output_md=out_md,
        )

    # Once all 5 exist, report builds cleanly
    tiers_yaml.write_text(
        yaml.safe_dump({"model_version": "v1_test", "date": "2026-11-11", "set_name": "primary", "thresholds": {"p90": 0.1, "p97": 0.2, "p99_5": 0.4}}),
        encoding="utf-8",
    )
    text = build_model_v1_report(
        results_csv=results_csv,
        ablation_csv=ablation_csv,
        breakdown_csv=breakdown_csv,
        importance_csv=importance_csv,
        tiers_yaml=tiers_yaml,
        output_md=out_md,
    )
    assert out_md.exists()
    assert "Model v1 Evaluation Report" in text
    assert "P99.5" in text
    assert "Pre-monsoon" in text
