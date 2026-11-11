"""Tests for Model v0 SHAP labels, additivity, local top-5 schema, and validation-only scope."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.explain.labels import FEATURE_LABELS, get_feature_label
from src.explain.shap_v0 import (
    SHAP_GLOBAL_V0_CSV,
    SHAP_LOCAL_V0_PATH,
    SHAP_SUMMARY_V0_PNG,
    assert_val_only_rows,
    compute_tree_shap_margins,
    extract_top_k_local_shap,
    verify_shap_additivity,
)
from src.models.dev_predictions import MODEL_V0_PATH, PREDICTIONS_V0_PATH
from src.models.features_io import FEATURES_FIRE_BASE_PATH, load_features
from src.models.train_v0 import V0_FEATURES


def test_every_v0_feature_has_label():
    for feat in V0_FEATURES:
        assert feat in FEATURE_LABELS
        lbl = get_feature_label(feat)
        assert isinstance(lbl, str) and len(lbl.strip()) > 0


def test_synthetic_shap_additivity_and_top_k_ranking():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(80, len(V0_FEATURES))).astype(np.float32), columns=V0_FEATURES)
    y = ((X["fire_count_7d"] + 0.5 * X["neighbour_fire_7d"] - 0.3 * X["lat_c"]) > 0).astype(int)

    clf = xgb.XGBClassifier(n_estimators=12, max_depth=3, random_state=42, tree_method="hist")
    clf.fit(X, y)

    shap_vals, base_val = compute_tree_shap_margins(clf, X)
    max_diff = verify_shap_additivity(clf, X, shap_vals, base_val, tol=1e-4)
    assert max_diff <= 1e-4

    keys = pd.DataFrame(
        {
            "cell_id": [f"r00c{i:02d}" for i in range(len(X))],
            "date": pd.date_range("2023-01-01", periods=len(X), freq="D"),
            "split": ["val"] * len(X),
        }
    )
    assert_val_only_rows(keys, "synthetic val keys")

    local_df = extract_top_k_local_shap(keys, X, shap_vals, V0_FEATURES, model_version="v0_test", k=5)
    assert len(local_df) == len(X) * 5
    assert list(local_df.columns) == ["cell_id", "date", "rank", "feature", "label", "value", "shap", "model_version"]
    assert not local_df.duplicated(subset=["cell_id", "date", "rank"]).any()
    assert set(local_df["rank"].unique()) == {1, 2, 3, 4, 5}
    for feat in local_df["feature"].unique():
        assert feat in FEATURE_LABELS

    # Verify descending |shap| order within each cell-day
    for _, grp in local_df.groupby(["cell_id", "date"], sort=False):
        grp_sorted = grp.sort_values("rank")
        assert list(grp_sorted["rank"]) == [1, 2, 3, 4, 5]
        abs_shaps = np.abs(grp_sorted["shap"].to_numpy())
        assert np.all(np.diff(abs_shaps) <= 1e-6)


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
def test_model_v0_1000_random_rows_additivity():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_V0_PATH)

    X_va, _, keys_va = load_features("val", label="new_fire_next_48h", feature_names=V0_FEATURES, primary=True)
    assert_val_only_rows(keys_va, "real val sample")

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(X_va), size=1000, replace=False))
    X_sub = X_va.iloc[idx][V0_FEATURES]

    shap_vals, base_val = compute_tree_shap_margins(model, X_sub)
    max_diff = verify_shap_additivity(model, X_sub, shap_vals, base_val, tol=1e-4)
    assert max_diff <= 1e-4


@pytest.mark.skipif(
    not (SHAP_LOCAL_V0_PATH.exists() and PREDICTIONS_V0_PATH.exists()),
    reason="Model v0 local SHAP parquet missing in CI",
)
def test_local_shap_v0_parquet_and_global_artifacts():
    shap_df = pd.read_parquet(SHAP_LOCAL_V0_PATH)
    preds_df = pd.read_parquet(PREDICTIONS_V0_PATH)

    expected_cols = ["cell_id", "date", "rank", "feature", "label", "value", "shap", "model_version"]
    assert list(shap_df.columns) == expected_cols
    assert not shap_df.duplicated(subset=["cell_id", "date", "rank"]).any()
    assert set(shap_df["rank"].unique()) == {1, 2, 3, 4, 5}

    # Every row must map to a valid feature label
    for feat, lbl in zip(shap_df["feature"], shap_df["label"]):
        assert feat in FEATURE_LABELS
        assert lbl == FEATURE_LABELS[feat]

    # Validation rows only (strictly 2023, zero train/test rows, exact match to High/Critical)
    assert_val_only_rows(shap_df, "local shap parquet")
    hc_preds = preds_df[preds_df["tier"].isin(["High", "Critical"])].copy()
    assert len(shap_df) == len(hc_preds) * 5

    shap_pairs = set(zip(shap_df["cell_id"].astype(str), pd.to_datetime(shap_df["date"])))
    pred_pairs = set(zip(hc_preds["cell_id"].astype(str), pd.to_datetime(hc_preds["date"])))
    assert shap_pairs == pred_pairs

    # Verify global CSV and summary PNG
    assert SHAP_GLOBAL_V0_CSV.exists()
    assert SHAP_SUMMARY_V0_PNG.exists()
    g_df = pd.read_csv(SHAP_GLOBAL_V0_CSV)
    assert list(g_df.columns) == ["feature", "mean_abs_shap", "rank"]
    assert len(g_df) == len(V0_FEATURES)
    assert list(g_df["rank"]) == [1, 2, 3, 4, 5, 6, 7]

