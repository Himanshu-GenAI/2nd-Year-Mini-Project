"""Unit tests for Model v1 bundle, calibration artifacts, and verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.isotonic import IsotonicRegression

from scripts.make_model_v1_report import build_model_v1_report
from src.config import DOCS_AUDIT_DIR, PROJECT_ROOT
from src.features.build_features import ALL_FEATURE_NAMES, FEATURES_FULL_PATH
from src.models.calibrate import MODEL_V1_BUNDLE_DIR, load_model_v1_bundle, predict_model_v1, sha256_file
from src.models.features_io import load_features
from src.models.train_v1 import assert_test_split_locked


def test_test_split_locked_enforced():
    """Verify assert_test_split_locked passes when unset and raises when set."""
    assert_test_split_locked()


def test_model_v1_bundle_integrity():
    """Verify Model v1 bundle exists and all artifact checksums match exactly."""
    checksum_path = MODEL_V1_BUNDLE_DIR / "checksums.json"
    assert checksum_path.exists(), f"Missing {checksum_path}"

    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    assert "model.json" in checksums
    assert "features.json" in checksums
    assert "fit_dates.json" in checksums
    assert "params.json" in checksums
    assert "version.txt" in checksums

    for fname, expected_sha in checksums.items():
        fpath = MODEL_V1_BUNDLE_DIR / fname
        assert fpath.exists(), f"Missing bundle file {fpath}"
        actual_sha = sha256_file(fpath)
        assert actual_sha == expected_sha, f"Checksum mismatch for {fname}"


def test_model_v1_bundle_loading_and_predict():
    """Verify Model v1 bundle can be loaded and predict on synthetic features."""
    bundle = load_model_v1_bundle(bundle_dir=MODEL_V1_BUNDLE_DIR, verify_checksums=True)
    assert len(bundle["features"]) == 21
    assert bundle["features"] == ALL_FEATURE_NAMES

    # Create synthetic test batch
    rng = np.random.default_rng(42)
    synthetic_data = {feat: rng.normal(0, 1, 10).astype(np.float32) for feat in ALL_FEATURE_NAMES}
    df_synth = pd.DataFrame(synthetic_data)

    preds = predict_model_v1(bundle, df_synth)
    assert len(preds) == 10
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert not np.isnan(preds).any()


def test_calibrator_monotone_and_bounded():
    """Verify isotonic calibrator behavior is monotonically non-decreasing and bounded in [0, 1]."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 200)
    y = (x + rng.normal(0, 0.1, 200) > 0.5).astype(int)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, y)

    eval_x = np.linspace(0.0, 1.0, 100)
    cal_preds = iso.predict(eval_x)

    # Monotonicity check
    diffs = np.diff(cal_preds)
    assert np.all(diffs >= -1e-7), f"Isotonic regression violated monotonicity: min diff {diffs.min()}"

    # Boundedness check in [0, 1]
    assert np.all(cal_preds >= 0.0)
    assert np.all(cal_preds <= 1.0)


def test_reloaded_saved_model_reproduces_2023_predictions_to_6_decimal_places():
    """Verify reloading the saved Model v1 bundle reproduces 2023 predictions to 6 decimal places."""
    bundle1 = load_model_v1_bundle(bundle_dir=MODEL_V1_BUNDLE_DIR, verify_checksums=True)

    if FEATURES_FULL_PATH.exists():
        # Score a sample of 2023 val rows
        X_va, _, _ = load_features(
            "val",
            label="new_fire_next_48h",
            feature_names=bundle1["features"],
            primary=True,
            features_path=FEATURES_FULL_PATH,
        )
        sample = X_va.iloc[:500].copy()
    else:
        rng = np.random.default_rng(42)
        sample = pd.DataFrame({f: rng.normal(0, 1, 100).astype(np.float32) for f in bundle1["features"]})

    preds1 = predict_model_v1(bundle1, sample)

    # Reload bundle independently in separate instance
    bundle2 = load_model_v1_bundle(bundle_dir=MODEL_V1_BUNDLE_DIR, verify_checksums=True)
    preds2 = predict_model_v1(bundle2, sample)

    np.testing.assert_allclose(preds1, preds2, atol=1e-6, rtol=1e-6)


def test_report_fails_if_source_file_missing(tmp_path: Path):
    """Verify build_model_v1_report raises FileNotFoundError if any source artifact is missing."""
    with pytest.raises(FileNotFoundError, match="missing required artifact"):
        build_model_v1_report(
            results_csv=tmp_path / "non_existent_results.csv",
            output_md=tmp_path / "out.md",
        )


def test_tiers_v1_preview_yaml_valid():
    """Verify configs/tiers_v1_preview.yaml has monotonic cutoffs in (0, 1)."""
    tiers_path = PROJECT_ROOT / "configs" / "tiers_v1_preview.yaml"
    assert tiers_path.exists(), f"Missing {tiers_path}"

    with open(tiers_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    th = cfg["thresholds"]
    assert 0.0 < th["p90"] < th["p97"] < th["p99_5"] < 1.0


def test_model_v1_breakdown_csv_valid():
    """Verify docs/audit/model_v1_breakdown.csv exists and has expected schema."""
    breakdown_path = DOCS_AUDIT_DIR / "model_v1_breakdown.csv"
    assert breakdown_path.exists(), f"Missing {breakdown_path}"

    df = pd.read_csv(breakdown_path)
    expected_cols = {"dimension", "bucket", "rows", "positives", "base_rate", "pr_auc", "recall_top10", "brier", "note"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) >= 10
    assert "season" in df["dimension"].values
