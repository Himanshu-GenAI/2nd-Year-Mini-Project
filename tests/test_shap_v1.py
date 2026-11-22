"""Unit and regression tests for Model v1 SHAP pipeline and artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explain.labels import FEATURE_LABELS, GROUPS, get_driver_group, get_feature_label
from src.explain.shap_v0 import (
    assert_val_only_rows,
    compute_tree_shap_margins,
    group_shap_values,
    verify_shap_additivity,
)
from src.explain.shap_v1 import (
    SHAP_GLOBAL_V1_FEATURES_CSV,
    SHAP_GLOBAL_V1_GROUPS_CSV,
    SHAP_LOCAL_V1_BY_FEATURE_PATH,
    SHAP_LOCAL_V1_PATH,
    SHAP_SANITY_V1_MD,
    SHAP_SUMMARY_V1_PNG,
)
from src.features.build_features import ALL_FEATURE_NAMES
from src.models.calibrate import MODEL_V1_BUNDLE_DIR, load_model_v1_bundle
from src.models.features_io import FEATURES_FULL_PATH, load_features
from src.models.train_v1 import assert_test_split_locked

PREDICTIONS_V1_PATH = Path("data/processed/predictions/model_v1_dev.parquet")


def test_every_feature_belongs_to_exactly_one_group():
    """Verify all 21 Model v1 features have human-readable labels and map to exactly 1 group."""
    assert len(ALL_FEATURE_NAMES) == 21
    for feat in ALL_FEATURE_NAMES:
        assert feat in FEATURE_LABELS, f"Missing label for feature {feat}"
        assert feat in GROUPS, f"Missing driver group for feature {feat}"
        lbl = get_feature_label(feat)
        grp = get_driver_group(feat)
        assert isinstance(lbl, str) and len(lbl.strip()) > 0
        assert isinstance(grp, str) and len(grp.strip()) > 0

    # Exactly 8 distinct driver groups
    unique_groups = {GROUPS[f] for f in ALL_FEATURE_NAMES}
    assert len(unique_groups) == 8, f"Expected 8 unique driver groups, found {len(unique_groups)}: {unique_groups}"
    expected_8_groups = {
        "Fires in this cell, last 7 days",
        "Fires in this cell, last 30 days",
        "Fires in nearby cells, last 7 days",
        "Regional baseline",
        "Seasonal baseline",
        "Recent heat",
        "Recent rainfall",
        "Drought index",
    }
    assert unique_groups == expected_8_groups


@pytest.mark.skipif(
    not (MODEL_V1_BUNDLE_DIR.exists() and FEATURES_FULL_PATH.exists()),
    reason="Model v1 bundle or features.parquet missing",
)
def test_model_v1_1000_random_rows_grouped_additivity():
    """Verify additivity within 1e-4 on 1,000 random rows and on grouped sums."""
    assert_test_split_locked()
    bundle = load_model_v1_bundle(bundle_dir=MODEL_V1_BUNDLE_DIR, verify_checksums=True)
    model = bundle["model"]
    feat_names = bundle["features"]

    X_va, _, keys_va = load_features("val", label="new_fire_next_48h", feature_names=feat_names, primary=True)
    assert_val_only_rows(keys_va, "real val sample")

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(X_va), size=1000, replace=False))
    X_sub = X_va.iloc[idx][feat_names]

    shap_vals, base_val = compute_tree_shap_margins(model, X_sub)
    # Raw feature additivity
    max_diff_raw = verify_shap_additivity(model, X_sub, shap_vals, base_val, tol=1e-4)
    assert max_diff_raw <= 1e-4

    # Grouped additivity
    _, grouped_shap, _ = group_shap_values(shap_vals, feat_names, GROUPS)
    max_diff_grouped = verify_shap_additivity(model, X_sub, grouped_shap, base_val, tol=1e-4)
    assert max_diff_grouped <= 1e-4


@pytest.mark.skipif(
    not (SHAP_LOCAL_V1_PATH.exists() and SHAP_LOCAL_V1_BY_FEATURE_PATH.exists() and PREDICTIONS_V1_PATH.exists()),
    reason="Model v1 SHAP parquet files missing",
)
def test_grouped_local_and_audit_shap_v1_artifacts():
    """Verify Model v1 grouped and audit local SHAP parquet files and schemas."""
    shap_df = pd.read_parquet(SHAP_LOCAL_V1_PATH)
    audit_df = pd.read_parquet(SHAP_LOCAL_V1_BY_FEATURE_PATH)
    preds_df = pd.read_parquet(PREDICTIONS_V1_PATH)

    # 1. Column names
    expected_cols = ["cell_id", "date", "rank", "driver", "shap", "feature_detail", "model_version"]
    assert list(shap_df.columns) == expected_cols

    # 2. Scope: Strictly 2023 validation primary rows
    assert_val_only_rows(shap_df, "grouped local shap v1")
    assert_val_only_rows(audit_df, "audit local shap v1")

    # 3. Row count check: exactly 5 * count of High and Critical cell-days
    hc_preds = preds_df[preds_df["tier"].isin(["High", "Critical"])].copy()
    n_hc = len(hc_preds)
    assert len(shap_df) == n_hc * 5
    assert len(audit_df) == n_hc * 5

    # 4. Unique driver per (cell_id, date)
    assert not shap_df.duplicated(subset=["cell_id", "date", "driver"]).any(), "Duplicate driver found per (cell_id, date)"

    # 5. Unique rank per (cell_id, date)
    assert not shap_df.duplicated(subset=["cell_id", "date", "rank"]).any(), "Duplicate rank found per (cell_id, date)"

    # 6. Ranks 1 to 5 strictly
    assert set(shap_df["rank"].unique()) == {1, 2, 3, 4, 5}

    # 7. Drivers are valid subset of 8 groups
    assert set(shap_df["driver"].unique()).issubset(set(GROUPS.values()))

    # 8. Feature details non-empty
    assert (shap_df["feature_detail"].astype(str).str.len() > 0).all()

    # 9. Cell-days match predictions cell-days exactly
    shap_pairs = set(zip(shap_df["cell_id"].astype(str), pd.to_datetime(shap_df["date"])))
    pred_pairs = set(zip(hc_preds["cell_id"].astype(str), pd.to_datetime(hc_preds["date"])))
    assert shap_pairs == pred_pairs

    # 10. Verify audit files exist
    assert SHAP_GLOBAL_V1_FEATURES_CSV.exists()
    assert SHAP_GLOBAL_V1_GROUPS_CSV.exists()
    assert SHAP_SUMMARY_V1_PNG.exists()
    assert SHAP_SANITY_V1_MD.exists()

    # 11. Verify global groups CSV schema and count (8 groups)
    grp_df = pd.read_csv(SHAP_GLOBAL_V1_GROUPS_CSV)
    assert list(grp_df.columns) == ["driver", "mean_abs_shap", "rank"]
    assert len(grp_df) == 8
    assert list(grp_df["rank"]) == list(range(1, 9))
