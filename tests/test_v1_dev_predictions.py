"""Unit tests for Model v1 development predictions and schemas."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.dev_predictions import assign_tiers, compute_tier_thresholds

PREDICTIONS_V1_PATH = PROCESSED_DIR / "predictions" / "model_v1_dev.parquet"
TIERS_V1_PATH = PROJECT_ROOT / "configs" / "tiers_v1_preview.yaml"


def test_tier_assignment_v1_synthetic():
    """Verify tier assignment maps to expected 90/7/2.5/0.5 shares on synthetic distribution."""
    scores = np.linspace(0.001, 0.999, 1000, dtype=np.float32)
    thresholds = compute_tier_thresholds(scores)
    assert thresholds["p90"] < thresholds["p97"] < thresholds["p99_5"]

    tiers = assign_tiers(scores, thresholds)
    shares = tiers.value_counts(normalize=True) * 100.0
    assert abs(shares["Low"] - 90.0) < 0.5
    assert abs(shares["Moderate"] - 7.0) < 0.5
    assert abs(shares["High"] - 2.5) < 0.5
    assert abs(shares["Critical"] - 0.5) < 0.5


def test_tiers_v1_preview_yaml_structure():
    """Verify configs/tiers_v1_preview.yaml has expected structure and valid thresholds."""
    assert TIERS_V1_PATH.exists(), f"Missing {TIERS_V1_PATH}"
    with open(TIERS_V1_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "model_version" in cfg
    assert re.match(r"^v1_.*", str(cfg["model_version"]))
    th = cfg["thresholds"]
    assert 0.0 < th["p90"] < th["p97"] < th["p99_5"] < 1.0


@pytest.mark.skipif(not PREDICTIONS_V1_PATH.exists(), reason="model_v1_dev.parquet not present")
def test_model_v1_dev_predictions_parquet():
    """Verify model_v1_dev.parquet schema, row count, uniqueness, and tier distributions."""
    df = pd.read_parquet(PREDICTIONS_V1_PATH)

    expected_cols = ["cell_id", "date", "score", "tier", "model_version", "split"]
    assert list(df.columns) == expected_cols
    assert len(df) == 373605

    # Unique (cell_id, date)
    assert not df.duplicated(subset=["cell_id", "date"]).any()

    # Scores strictly in [0, 1], zero NaNs
    assert df["score"].isna().sum() == 0
    assert (df["score"] >= 0.0).all() and (df["score"] <= 1.0).all()

    # Only validation split (2023), no train/test/gap
    assert set(df["split"].unique()) == {"val"}
    years = pd.to_datetime(df["date"]).dt.year.unique()
    assert list(years) == [2023]

    # Model version format
    versions = df["model_version"].unique()
    assert len(versions) == 1
    assert str(versions[0]).startswith("v1_")

    # Tier shares near 90 / 7 / 2.5 / 0.5 percent
    shares = df["tier"].value_counts(normalize=True) * 100.0
    assert abs(shares["Low"] - 90.0) < 0.25
    assert abs(shares["Moderate"] - 7.0) < 0.25
    assert abs(shares["High"] - 2.5) < 0.25
    assert abs(shares["Critical"] - 0.5) < 0.25
