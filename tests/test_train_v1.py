"""Unit tests for Model v1 training, tuning, RF benchmark, and selection logic (Day 5)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import ALL_FEATURE_NAMES
from src.models import train_v1
from src.models.train_v0 import V0_FEATURES
from src.models.train_v1 import (
    FEATURE_GROUPS,
    assert_test_split_locked,
    compute_feature_importance,
    fit_xgb_two_stage,
    load_xgb_model,
    run_random_forest,
    run_xgb_tuning,
    select_winner,
)


def _make_synthetic_dataset(n_cells: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    cells = [f"r00c{i:02d}" for i in range(n_cells)]

    def _build_split(dates: pd.DatetimeIndex, split_name: str):
        rows = []
        for c in cells:
            for d in dates:
                rows.append({"cell_id": c, "date": d, "split": split_name, "burning_prev3": False})
        keys = pd.DataFrame(rows)
        n = len(keys)
        X = pd.DataFrame(
            rng.normal(0.0, 1.0, size=(n, len(ALL_FEATURE_NAMES))).astype(np.float32),
            columns=ALL_FEATURE_NAMES,
        )
        # Deterministic signal combining G1, G2, and G3 features
        logit = -1.6 + 1.2 * X["fire_count_7d"] + 0.9 * X["tmax_t0"] - 0.7 * X["precip_sum_7d"] + 0.8 * X["kbdi"]
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = pd.Series((rng.random(n) < prob).astype(np.int64), name="new_fire_next_48h")
        return X, y, keys

    tr_dates = pd.date_range("2020-01-01", "2022-12-29", freq="10D")
    va_dates = pd.date_range("2023-01-01", "2023-12-29", freq="10D")
    X_tr, y_tr, keys_tr = _build_split(tr_dates, "train")
    X_va, y_va, keys_va = _build_split(va_dates, "val")
    return X_tr, y_tr, keys_tr, X_va, y_va, keys_va


def test_feature_groups_and_test_split_lock(monkeypatch):
    assert FEATURE_GROUPS["G1"] == V0_FEATURES
    assert len(FEATURE_GROUPS["G1"]) == 7
    assert len(FEATURE_GROUPS["G2"]) == 19
    assert len(FEATURE_GROUPS["G3"]) == 21
    assert set(FEATURE_GROUPS["G1"]) < set(FEATURE_GROUPS["G2"]) < set(FEATURE_GROUPS["G3"])
    assert set(FEATURE_GROUPS["G3"]) == set(ALL_FEATURE_NAMES)

    # Ensure no test split loading exists in train_v1 or scripts/run_tuning.py
    src_train = inspect.getsource(train_v1)
    assert 'load_features(\n            "test"' not in src_train
    assert 'load_features("test"' not in src_train
    assert "split == \"test\"" not in src_train
    run_tuning_text = Path("scripts/run_tuning.py").read_text(encoding="utf-8")
    assert 'load_features(\n        "test"' not in run_tuning_text
    assert 'load_features("test"' not in run_tuning_text

    monkeypatch.setenv("ALLOW_TEST_SPLIT", "1")
    with pytest.raises(RuntimeError, match="ALLOW_TEST_SPLIT"):
        assert_test_split_locked()
    monkeypatch.delenv("ALLOW_TEST_SPLIT", raising=False)
    assert_test_split_locked()


def test_two_stage_xgb_determinism_and_leakage_guards():
    X_tr, y_tr, keys_tr, X_va, _, _ = _make_synthetic_dataset(seed=42)

    _, n1, s1 = fit_xgb_two_stage(
        X_tr, y_tr, keys_tr["date"], X_va, max_trees=40, early_stopping_rounds=10, seed=42, n_jobs=1
    )
    _, n2, s2 = fit_xgb_two_stage(
        X_tr, y_tr, keys_tr["date"], X_va, max_trees=40, early_stopping_rounds=10, seed=42, n_jobs=1
    )
    assert n1 == n2
    np.testing.assert_allclose(s1, s2, rtol=0, atol=1e-7)

    # Leakage guard: dates > 2022 in train_dates must raise ValueError
    bad_dates = keys_tr["date"].copy()
    bad_dates.iloc[-1] = pd.Timestamp("2023-01-05")
    with pytest.raises(ValueError, match="2022"):
        fit_xgb_two_stage(X_tr, y_tr, bad_dates, X_va, max_trees=20)

    # Guard: missing 2022 early-stopping slice must raise ValueError
    mask_no_2022 = pd.to_datetime(keys_tr["date"]).dt.year <= 2021
    with pytest.raises(ValueError, match="2022"):
        fit_xgb_two_stage(X_tr[mask_no_2022], y_tr[mask_no_2022], keys_tr.loc[mask_no_2022, "date"], X_va, max_trees=20)


def test_tuning_random_search_and_save_reload(tmp_path: Path):
    X_tr, y_tr, keys_tr, X_va, y_va, keys_va = _make_synthetic_dataset(seed=42)
    runs_csv = tmp_path / "runs.csv"
    res_csv = tmp_path / "results.csv"
    res_md = tmp_path / "results.md"
    model_path = tmp_path / "xgb_v1_tuned" / "model.json"

    model, best_cfg, trials_df, scores = run_xgb_tuning(
        X_train=X_tr,
        y_train=y_tr,
        train_dates=keys_tr["date"],
        X_val=X_va,
        y_val=y_va,
        val_keys=keys_va,
        feature_group_name="G3",
        n_trials=4,
        max_minutes=5.0,
        max_trees=30,
        seed=42,
        runs_csv_path=runs_csv,
        results_csv_path=res_csv,
        results_md_path=res_md,
        model_save_path=model_path,
    )

    assert len(trials_df) == 4
    assert "best_trees" in best_cfg
    assert model_path.exists()
    assert runs_csv.exists()
    assert res_csv.exists()

    reloaded = load_xgb_model(model_path)
    np.testing.assert_allclose(reloaded.predict_proba(X_va)[:, 1], scores, rtol=1e-6, atol=1e-6)


def test_select_winner_rules():
    # Case 1: Strict >0.005 PR-AUC improvement wins regardless of recall difference
    win_clear = select_winner(
        [
            {"model": "a", "pr_auc": 0.320, "recall_top10": 0.60, "complexity": 2},
            {"model": "b", "pr_auc": 0.310, "recall_top10": 0.80, "complexity": 1},
        ]
    )
    assert win_clear["model"] == "a"

    # Case 2: Within 0.005 PR-AUC -> higher recall@10% wins
    win_recall = select_winner(
        [
            {"model": "xgb_tuned", "pr_auc": 0.314, "recall_top10": 0.68, "complexity": 2},
            {"model": "rf_v1", "pr_auc": 0.311, "recall_top10": 0.74, "complexity": 3},
        ]
    )
    assert win_recall["model"] == "rf_v1"

    # Case 3: Within 0.005 PR-AUC AND tied Recall@10% -> simpler model (lower complexity) wins
    win_simple = select_winner(
        [
            {"model": "xgb_default", "pr_auc": 0.312, "recall_top10": 0.70, "complexity": 1},
            {"model": "xgb_tuned", "pr_auc": 0.314, "recall_top10": 0.70, "complexity": 2},
        ]
    )
    assert win_simple["model"] == "xgb_default"


def test_rf_and_feature_importance_synthetic(tmp_path: Path):
    X_tr, y_tr, _, X_va, y_va, keys_va = _make_synthetic_dataset(seed=42)
    runs_csv = tmp_path / "runs.csv"
    res_csv = tmp_path / "results.csv"
    res_md = tmp_path / "results.md"
    fi_csv = tmp_path / "feature_importance.csv"

    rf, res, scores = run_random_forest(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        val_keys=keys_va,
        n_estimators=10,
        min_samples_leaf=2,
        max_samples=0.8,
        seed=42,
        runs_csv_path=runs_csv,
        results_csv_path=res_csv,
        results_md_path=res_md,
    )
    assert len(scores) == len(X_va)
    assert res["model"] == "rf_v1"

    imp_df = compute_feature_importance(
        rf,
        X_va,
        y_va,
        feature_names=ALL_FEATURE_NAMES,
        n_sample=200,
        n_repeats=2,
        seed=42,
        output_path=fi_csv,
    )
    assert fi_csv.exists()
    assert len(imp_df) == len(ALL_FEATURE_NAMES)
    assert set(imp_df.columns) == {"feature", "gain_importance", "perm_importance_mean", "perm_importance_std"}
    assert imp_df["perm_importance_mean"].is_monotonic_decreasing

