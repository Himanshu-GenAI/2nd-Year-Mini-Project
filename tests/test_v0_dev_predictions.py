import re

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.dev_predictions import (
    PREDICTIONS_V0_PATH,
    TIERS_PREVIEW_V0_PATH,
    assign_tiers,
    compute_tier_thresholds,
)


def test_tier_assignment_synthetic():
    scores = np.linspace(0.001, 0.999, 1000, dtype=np.float32)
    thresholds = compute_tier_thresholds(scores)
    assert thresholds["p90"] < thresholds["p97"] < thresholds["p99_5"]

    tiers = assign_tiers(scores, thresholds)
    shares = tiers.value_counts(normalize=True) * 100.0
    assert abs(shares["Low"] - 90.0) < 0.5
    assert abs(shares["Moderate"] - 7.0) < 0.5
    assert abs(shares["High"] - 2.5) < 0.5
    assert abs(shares["Critical"] - 0.5) < 0.5


def test_tiers_preview_yaml_valid():
    assert TIERS_PREVIEW_V0_PATH.exists(), f"Missing {TIERS_PREVIEW_V0_PATH}"
    with open(TIERS_PREVIEW_V0_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert re.match(r"^v0_plain_\d{8}$", str(cfg["model_version"]))
    th = cfg["thresholds"]
    assert 0.0 < th["p90"] < th["p97"] < th["p99_5"] < 1.0
    assert abs(cfg["val_pr_auc"] - 0.2478) <= 0.0005


@pytest.mark.skipif(not PREDICTIONS_V0_PATH.exists(), reason="Dev predictions file not present in CI")
def test_model_v0_dev_predictions_parquet():
    df = pd.read_parquet(PREDICTIONS_V0_PATH)

    expected_cols = ["cell_id", "date", "score", "tier", "model_version", "split"]
    assert list(df.columns) == expected_cols
    assert len(df) == 373605

    # Unique (cell_id, date)
    assert not df.duplicated(subset=["cell_id", "date"]).any()

    # Scores strictly in [0, 1], no NaNs
    assert df["score"].isna().sum() == 0
    assert (df["score"] >= 0.0).all() and (df["score"] <= 1.0).all()

    # Only validation split (2023), no train/test/gap
    assert set(df["split"].unique()) == {"val"}
    years = pd.to_datetime(df["date"]).dt.year.unique()
    assert list(years) == [2023]

    # Model version format
    versions = df["model_version"].unique()
    assert len(versions) == 1
    assert re.match(r"^v0_plain_\d{8}$", str(versions[0]))

    # Tier shares near 90 / 7 / 2.5 / 0.5 percent
    shares = df["tier"].value_counts(normalize=True) * 100.0
    assert abs(shares["Low"] - 90.0) < 0.25
    assert abs(shares["Moderate"] - 7.0) < 0.25
    assert abs(shares["High"] - 2.5) < 0.25
    assert abs(shares["Critical"] - 0.5) < 0.25

