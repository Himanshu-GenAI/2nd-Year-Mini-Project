"""Tests verifying API and dashboard schema compatibility between Model v0 and Model v1."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import GRID_PATH, PROCESSED_DIR
from src.explain.labels import GROUPS
from src.features.build_features import ALL_FEATURE_NAMES

PRED_V0_PATH = PROCESSED_DIR / "predictions" / "model_v0_dev.parquet"
PRED_V1_PATH = PROCESSED_DIR / "predictions" / "model_v1_dev.parquet"
SHAP_V0_PATH = PROCESSED_DIR / "shap" / "model_v0_dev.parquet"
SHAP_V1_PATH = PROCESSED_DIR / "shap" / "model_v1_dev.parquet"
SHAP_BY_FEAT_V0_PATH = PROCESSED_DIR / "shap" / "model_v0_dev_by_feature.parquet"
SHAP_BY_FEAT_V1_PATH = PROCESSED_DIR / "shap" / "model_v1_dev_by_feature.parquet"


@pytest.mark.skipif(not GRID_PATH.exists(), reason="Grid file not present in CI")
def test_grid_file_schema():
    """Verify spatial grid definition file exists and has required columns."""
    assert GRID_PATH.exists(), f"Missing grid file at {GRID_PATH}"
    df_grid = pd.read_parquet(GRID_PATH)
    expected_cols = {"cell_id", "lat_c", "lon_c", "row", "col", "state"}
    assert expected_cols.issubset(set(df_grid.columns))
    assert len(df_grid) == 1167
    assert df_grid["cell_id"].nunique() == 1167


@pytest.mark.skipif(
    not (PRED_V0_PATH.exists() and PRED_V1_PATH.exists()),
    reason="Predictions parquet files missing",
)
def test_predictions_schema_compatibility():
    """Verify Model v1 predictions schema, column names, and dtypes match Model v0 exactly."""
    df0 = pd.read_parquet(PRED_V0_PATH)
    df1 = pd.read_parquet(PRED_V1_PATH)

    # Column names must match exactly
    assert list(df0.columns) == list(df1.columns)
    expected_cols = ["cell_id", "date", "score", "tier", "model_version", "split"]
    assert list(df1.columns) == expected_cols

    # Data types must match
    for col in expected_cols:
        t0 = df0[col].dtype
        t1 = df1[col].dtype
        if col == "tier":
            # Both are categoricals containing standard tiers
            assert str(t0).startswith("category")
            assert str(t1).startswith("category")
            assert set(df0["tier"].cat.categories) == {"Low", "Moderate", "High", "Critical"}
            assert set(df1["tier"].cat.categories) == {"Low", "Moderate", "High", "Critical"}
        else:
            assert t0 == t1, f"Dtype mismatch for {col}: v0 has {t0}, v1 has {t1}"

    # Row count must match primary set count
    assert len(df0) == len(df1) == 373605
    assert set(df1["split"].unique()) == {"val"}


@pytest.mark.skipif(
    not (SHAP_V0_PATH.exists() and SHAP_V1_PATH.exists()),
    reason="SHAP parquet files missing",
)
def test_shap_grouped_schema_compatibility():
    """Verify Model v1 grouped SHAP schema, column names, and dtypes match Model v0."""
    s0 = pd.read_parquet(SHAP_V0_PATH)
    s1 = pd.read_parquet(SHAP_V1_PATH)

    assert list(s0.columns) == list(s1.columns)
    expected_cols = ["cell_id", "date", "rank", "driver", "shap", "feature_detail", "model_version"]
    assert list(s1.columns) == expected_cols

    for col in expected_cols:
        t0 = s0[col].dtype
        t1 = s1[col].dtype
        assert t0 == t1, f"SHAP Dtype mismatch for {col}: v0 has {t0}, v1 has {t1}"

    # Verify rank ranges 1 to 5
    assert set(s1["rank"].unique()) == {1, 2, 3, 4, 5}

    # Verify new driver groups appear in v1
    v0_drivers = set(s0["driver"].unique())
    v1_drivers = set(s1["driver"].unique())
    assert v0_drivers.issubset(set(GROUPS.values()))
    assert v1_drivers.issubset(set(GROUPS.values()))

    # New weather/drought driver groups must be present in v1
    new_drivers = {"Recent heat", "Recent rainfall", "Drought index"}
    assert len(new_drivers.intersection(v1_drivers)) > 0, "Expected new weather/drought driver groups in v1"


@pytest.mark.skipif(
    not (SHAP_BY_FEAT_V0_PATH.exists() and SHAP_BY_FEAT_V1_PATH.exists()),
    reason="Per-feature SHAP parquet files missing",
)
def test_shap_by_feature_schema_compatibility():
    """Verify Model v1 per-feature audit SHAP matches Model v0 schema and dtypes."""
    sf0 = pd.read_parquet(SHAP_BY_FEAT_V0_PATH)
    sf1 = pd.read_parquet(SHAP_BY_FEAT_V1_PATH)

    assert list(sf0.columns) == list(sf1.columns)
    expected_cols = ["cell_id", "date", "rank", "feature", "label", "value", "shap", "model_version"]
    assert list(sf1.columns) == expected_cols

    for col in expected_cols:
        t0 = sf0[col].dtype
        t1 = sf1[col].dtype
        assert t0 == t1, f"Audit SHAP Dtype mismatch for {col}: v0 has {t0}, v1 has {t1}"

    # Features in sf1 must be subset of the 21 features
    assert set(sf1["feature"].unique()).issubset(set(ALL_FEATURE_NAMES))
