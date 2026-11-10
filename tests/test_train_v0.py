from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import FIRE_TABLE_PATH
from src.models.features_io import load_features
from src.models.train_v0 import (
    V0_FEATURES,
    fit_and_score_logreg,
    fit_and_score_xgb_two_stage,
    prior_correct,
    subsample_negatives,
)
from src.utils.fit_registry import assert_fit_within_train, clear_registry, get_registered_fits


def test_prior_correction_properties():
    assert prior_correct(0.0, 0.2) == 0.0
    assert prior_correct(1.0, 0.2) == 1.0

    # beta=1 returns exact input
    p_vals = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_allclose(prior_correct(p_vals, beta=1.0), p_vals)

    # Strictly increasing for beta=0.2
    corr = prior_correct(p_vals, beta=0.2)
    assert (np.diff(corr) > 0).all()


def test_subsample_negatives_ratios():
    n = 10000
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    # 500 positives, 9500 negatives
    y = pd.Series([1] * 500 + [0] * 9500)

    X_sub, y_sub = subsample_negatives(X, y, beta=0.2, seed=42)
    # All positives retained
    assert int((y_sub == 1).sum()) == 500
    # Negatives within 1 percentage point of 20%
    kept_neg = int((y_sub == 0).sum())
    frac_neg = kept_neg / 9500.0
    assert abs(frac_neg - 0.20) <= 0.01


def test_logreg_scaler_fit_within_train_and_determinism():
    clear_registry()
    rng = np.random.default_rng(42)
    n_tr, n_va = 400, 100
    X_tr = pd.DataFrame(np.abs(rng.normal(size=(n_tr, len(V0_FEATURES)))), columns=V0_FEATURES)
    y_tr = pd.Series((rng.random(n_tr) < 0.2).astype(int))
    dates_tr = pd.Series(pd.date_range("2015-01-03", periods=n_tr, freq="D"))
    X_va = pd.DataFrame(np.abs(rng.normal(size=(n_va, len(V0_FEATURES)))), columns=V0_FEATURES)

    _, _, s1 = fit_and_score_logreg(X_tr, y_tr, dates_tr, X_va, seed=42)
    _, _, s2 = fit_and_score_logreg(X_tr, y_tr, dates_tr, X_va, seed=42)
    np.testing.assert_allclose(s1, s2)

    fits = [r for r in get_registered_fits() if r.name == "scaler_logreg_v0"]
    assert len(fits) > 0
    assert fits[-1].last_date <= date(2022, 12, 29)
    assert_fit_within_train(fits[-1])


def test_xgb_determinism_on_toy_data():
    rng = np.random.default_rng(42)
    dates_2021 = pd.date_range("2021-01-01", periods=100, freq="D")
    dates_2022 = pd.date_range("2022-01-01", periods=50, freq="D")
    all_dates = pd.Series(list(dates_2021) + list(dates_2022))

    n_tr = len(all_dates)
    X_tr = pd.DataFrame(rng.normal(size=(n_tr, len(V0_FEATURES))), columns=V0_FEATURES)
    y_tr = pd.Series((rng.random(n_tr) < 0.2).astype(int))
    X_va = pd.DataFrame(rng.normal(size=(40, len(V0_FEATURES))), columns=V0_FEATURES)

    _, t1, s1 = fit_and_score_xgb_two_stage(X_tr, y_tr, all_dates, X_va, variant="plain", seed=42)
    _, t2, s2 = fit_and_score_xgb_two_stage(X_tr, y_tr, all_dates, X_va, variant="plain", seed=42)
    assert t1 == t2
    np.testing.assert_allclose(s1, s2)


@pytest.mark.skipif(
    not FIRE_TABLE_PATH.exists() or not Path("data/processed/features_fire_base.parquet").exists(),
    reason="Fire table or features_fire_base not available",
)
def test_load_features_guards():
    with pytest.raises(PermissionError, match="locked"):
        load_features("test", "new_fire_next_48h", V0_FEATURES)

    with pytest.raises(ValueError, match="prohibited"):
        load_features("gap", "new_fire_next_48h", V0_FEATURES)

    X_va, y_va, keys_va = load_features("val", "new_fire_next_48h", V0_FEATURES, primary=True)
    assert list(X_va.columns) == V0_FEATURES
    assert not keys_va["burning_prev3"].any()
    assert len(X_va) == len(y_va) == len(keys_va)

