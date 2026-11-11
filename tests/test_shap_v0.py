"""Tests for grouped Model v0 SHAP labels, additivity, local/global schemas, and val-only scope."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.explain.labels import FEATURE_LABELS, GROUPS, get_driver_group, get_feature_label
from src.explain.shap_v0 import (
    SHAP_GLOBAL_V0_CSV,
    SHAP_GLOBAL_V0_GROUPS_CSV,
    SHAP_LOCAL_V0_BY_FEATURE_PATH,
    SHAP_LOCAL_V0_PATH,
    SHAP_SUMMARY_V0_PNG,
    assert_val_only_rows,
    compute_tree_shap_margins,
    extract_top_k_grouped_local_shap,
    group_shap_values,
    verify_shap_additivity,
)
from src.models.dev_predictions import MODEL_V0_PATH, PREDICTIONS_V0_PATH
from src.models.features_io import FEATURES_FIRE_BASE_PATH, load_features
from src.models.train_v0 import V0_FEATURES


def test_every_v0_feature_has_label_and_group():
    assert GROUPS["lat_c"] == "Regional baseline"
    assert GROUPS["lon_c"] == "Regional baseline"
    assert GROUPS["month_sin"] == "Seasonal baseline"
    assert GROUPS["month_cos"] == "Seasonal baseline"

    for feat in V0_FEATURES:
        assert feat in FEATURE_LABELS
        assert feat in GROUPS
        lbl = get_feature_label(feat)
        grp = get_driver_group(feat)
        assert isinstance(lbl, str) and len(lbl.strip()) > 0
        assert isinstance(grp, str) and len(grp.strip()) > 0
        if feat.startswith(("fire_count_", "neighbour_fire_")):
            assert grp == lbl

    assert len({GROUPS[f] for f in V0_FEATURES}) == 5


def test_synthetic_grouped_shap_additivity_and_top_k():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(80, len(V0_FEATURES))).astype(np.float32), columns=V0_FEATURES)
    y = ((X["fire_count_7d"] + 0.5 * X["neighbour_fire_7d"] - 0.3 * X["lat_c"]) > 0).astype(int)

    clf = xgb.XGBClassifier(n_estimators=12, max_depth=3, random_state=42, tree_method="hist")
    clf.fit(X, y)

    shap_vals, base_val = compute_tree_shap_margins(clf, X)
    driver_names, grouped_shap, _ = group_shap_values(shap_vals, V0_FEATURES)
    assert len(driver_names) == 5

    max_diff = verify_shap_additivity(clf, X, grouped_shap, base_val, tol=1e-4)
    assert max_diff <= 1e-4

    keys = pd.DataFrame(
        {
            "cell_id": [f"r00c{i:02d}" for i in range(len(X))],
            "date": pd.date_range("2023-01-01", periods=len(X), freq="D"),
            "split": ["val"] * len(X),
        }
    )
    assert_val_only_rows(keys, "synthetic val keys")

    n_groups = len({GROUPS[f] for f in V0_FEATURES})
    k_expected = min(5, n_groups)
    local_df = extract_top_k_grouped_local_shap(keys, X, shap_vals, V0_FEATURES, model_version="v0_test", k=5)

    assert len(local_df) == len(X) * k_expected
    assert list(local_df.columns) == ["cell_id", "date", "rank", "driver", "shap", "feature_detail", "model_version"]
    assert not local_df.duplicated(subset=["cell_id", "date", "rank"]).any()
    assert not local_df.duplicated(subset=["cell_id", "date", "driver"]).any()
    assert set(local_df["rank"].unique()) == set(range(1, k_expected + 1))
    assert set(local_df["driver"].unique()).issubset(set(GROUPS.values()))
    assert (local_df["feature_detail"].str.len() > 0).all()

    # Check Regional baseline feature_detail lists both lat_c and lon_c
    reg_details = local_df.loc[local_df["driver"] == "Regional baseline", "feature_detail"]
    assert len(reg_details) > 0
    assert reg_details.str.contains("lat_c=").all() and reg_details.str.contains("lon_c=").all()


def test_assert_val_only_rows_rejects_train_and_test():
    bad_train = pd.DataFrame({"cell_id": ["r00c00"], "date": [pd.Timestamp("2022-05-01")], "split": ["train"]})
    with pytest.raises(RuntimeError, match="Non-val split"):
        assert_val_only_rows(bad_train, "train test")

    bad_test = pd.DataFrame({"cell_id": ["r00c00"], "date": [pd.Timestamp("2024-05-01")], "split": ["test"]})
    with pytest.raises(RuntimeError, match="Non-val split"):
        assert_val_only_rows(bad_test, "test check")

    bad_year = pd.DataFrame({"cell_id": ["r00c00"], "date": [pd.Timestamp("2024-05-01")], "split": ["val"]})
    with pytest.raises(RuntimeError, match="Non-2023"):
        assert_val_only_rows(bad_year, "year check")


@pytest.mark.skipif(
    not (MODEL_V0_PATH.exists() and FEATURES_FIRE_BASE_PATH.exists()),
    reason="Model v0 or base features missing in CI",
)
def test_model_v0_1000_random_rows_grouped_additivity():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_V0_PATH)

    X_va, _, keys_va = load_features("val", label="new_fire_next_48h", feature_names=V0_FEATURES, primary=True)
    assert_val_only_rows(keys_va, "real val sample")

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(X_va), size=1000, replace=False))
    X_sub = X_va.iloc[idx][V0_FEATURES]

    shap_vals, base_val = compute_tree_shap_margins(model, X_sub)
    _, grouped_shap, _ = group_shap_values(shap_vals, V0_FEATURES)
    max_diff = verify_shap_additivity(model, X_sub, grouped_shap, base_val, tol=1e-4)
    assert max_diff <= 1e-4


@pytest.mark.skipif(
    not (SHAP_LOCAL_V0_PATH.exists() and SHAP_LOCAL_V0_BY_FEATURE_PATH.exists() and PREDICTIONS_V0_PATH.exists()),
    reason="Model v0 SHAP parquets missing in CI",
)
def test_grouped_local_and_audit_shap_v0_artifacts():
    shap_df = pd.read_parquet(SHAP_LOCAL_V0_PATH)
    audit_df = pd.read_parquet(SHAP_LOCAL_V0_BY_FEATURE_PATH)
    preds_df = pd.read_parquet(PREDICTIONS_V0_PATH)

    expected_cols = ["cell_id", "date", "rank", "driver", "shap", "feature_detail", "model_version"]
    assert list(shap_df.columns) == expected_cols
    assert not shap_df.duplicated(subset=["cell_id", "date", "rank"]).any()
    assert not shap_df.duplicated(subset=["cell_id", "date", "driver"]).any()

    n_groups = len({GROUPS[f] for f in V0_FEATURES})
    k_expected = min(5, n_groups)
    assert set(shap_df["rank"].unique()) == set(range(1, k_expected + 1))
    assert set(shap_df["driver"].unique()).issubset(set(GROUPS.values()))
    assert (shap_df["feature_detail"].astype(str).str.len() > 0).all()

    assert_val_only_rows(shap_df, "grouped local shap parquet")
    hc_preds = preds_df[preds_df["tier"].isin(["High", "Critical"])].copy()
    assert len(shap_df) == len(hc_preds) * k_expected
    assert len(audit_df) == len(hc_preds) * 5

    shap_pairs = set(zip(shap_df["cell_id"].astype(str), pd.to_datetime(shap_df["date"])))
    pred_pairs = set(zip(hc_preds["cell_id"].astype(str), pd.to_datetime(hc_preds["date"])))
    assert shap_pairs == pred_pairs

    # Verify audit per-feature table and global CSVs
    assert list(audit_df.columns) == ["cell_id", "date", "rank", "feature", "label", "value", "shap", "model_version"]
    assert SHAP_GLOBAL_V0_CSV.exists()
    assert SHAP_GLOBAL_V0_GROUPS_CSV.exists()
    assert SHAP_SUMMARY_V0_PNG.exists()

    grp_df = pd.read_csv(SHAP_GLOBAL_V0_GROUPS_CSV)
    assert list(grp_df.columns) == ["driver", "mean_abs_shap", "rank"]
    assert len(grp_df) == n_groups
    assert list(grp_df["rank"]) == list(range(1, n_groups + 1))
