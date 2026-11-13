"""Unit tests for partial weather diagnostics (bootstrap, rolling-origin, KBDI, fire season)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.diagnostics import (
    DIAGNOSTIC_FEATURE_GROUPS,
    evaluate_fire_season,
    get_calendar_weeks,
    run_block_bootstrap,
)
from src.models.train_v1 import FEATURE_GROUPS


def test_diagnostic_feature_groups_integrity():
    assert len(DIAGNOSTIC_FEATURE_GROUPS["G1"]) == 7
    assert len(DIAGNOSTIC_FEATURE_GROUPS["G1K"]) == 9
    assert len(DIAGNOSTIC_FEATURE_GROUPS["G2"]) == 19
    assert len(DIAGNOSTIC_FEATURE_GROUPS["G2R"]) == 18
    assert len(DIAGNOSTIC_FEATURE_GROUPS["G3"]) == 21

    # G1K is G1 plus KBDI
    assert set(DIAGNOSTIC_FEATURE_GROUPS["G1K"]) == set(FEATURE_GROUPS["G1"]) | {"kbdi", "kbdi_change_7d"}

    # G2R is G2 without rainfall sums and days_since_rain, but with KBDI
    g2_rain_removed = set(FEATURE_GROUPS["G2"]) - {"precip_sum_7d", "precip_sum_30d", "days_since_rain"}
    assert set(DIAGNOSTIC_FEATURE_GROUPS["G2R"]) == g2_rain_removed | {"kbdi", "kbdi_change_7d"}
    assert set(DIAGNOSTIC_FEATURE_GROUPS["G3"]) == set(FEATURE_GROUPS["G3"])


def test_get_calendar_weeks():
    dates = pd.date_range("2023-01-01", "2023-01-15", freq="D")
    weeks = get_calendar_weeks(pd.Series(dates))
    assert len(weeks) == 15
    # 2023-01-01 is Sunday (week ending 2023-01-01)
    # 2023-01-02 to 2023-01-08 is week ending 2023-01-08
    assert weeks.iloc[0] == "2022-12-26/2023-01-01"
    assert weeks.iloc[1] == "2023-01-02/2023-01-08"
    assert weeks.iloc[7] == "2023-01-02/2023-01-08"
    assert weeks.iloc[8] == "2023-01-09/2023-01-15"


def test_run_block_bootstrap_synthetic():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2023-12-29", freq="D")
    rows = []
    for d in dates:
        for c in range(5):
            rows.append({"cell_id": f"c{c}", "date": d})
    df = pd.DataFrame(rows)
    n = len(df)

    y = (rng.random(n) < 0.08).astype(int)
    # Give G2 slightly better signal than G1, and G3 comparable to G2
    s_clim = rng.random(n)
    s_g1 = s_clim + 0.5 * y + rng.normal(0, 0.5, n)
    s_g2 = s_g1 + 0.3 * y + rng.normal(0, 0.3, n)
    s_g3 = s_g2 + 0.05 * y + rng.normal(0, 0.1, n)

    df["label"] = y
    df["clim"] = s_clim
    df["g1"] = s_g1
    df["g2"] = s_g2
    df["g3"] = s_g3

    score_cols = {"climatology": "clim", "G1": "g1", "G2": "g2", "G3": "g3"}
    draws, summary = run_block_bootstrap(
        df=df,
        label_col="label",
        score_cols=score_cols,
        date_col="date",
        n_boot=50,
        seed=42,
    )
    assert len(draws) == 50
    assert set(draws.columns) == {
        "replicate",
        "diff_G2_G1",
        "diff_G3_G2",
        "diff_G3_clim",
        "pr_clim",
        "pr_G1",
        "pr_G2",
        "pr_G3",
    }
    assert len(summary) == 3
    assert set(summary["contrast"]) == {"G2 - G1", "G3 - G2", "G3 - climatology"}
    for _, row in summary.iterrows():
        assert row["ci_lower_95"] <= row["ci_upper_95"]
        assert row["includes_zero"] == (row["ci_lower_95"] <= 0.0 <= row["ci_upper_95"])


def test_evaluate_fire_season_synthetic():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2023-12-29", freq="5D")
    rows = []
    for d in dates:
        for c in range(4):
            rows.append({"cell_id": f"c{c}", "date": d})
    df = pd.DataFrame(rows)
    n = len(df)
    df["new_fire_next_48h"] = (rng.random(n) < 0.1).astype(int)
    df["score_climatology"] = rng.random(n)
    df["score_G1"] = rng.random(n)
    df["score_G2"] = rng.random(n)
    df["score_G3"] = rng.random(n)

    res = evaluate_fire_season(df, peak_months=[2, 3, 4], set_name="partial: 4 cells")
    assert len(res) == 4
    assert set(res["group"]) == {"climatology", "G1", "G2", "G3"}
    assert (res["season_months"] == "2-3-4").all()
    assert (res["set_name"] == "partial: 4 cells").all()
    assert (res["rows"] > 0).all()
    assert (res["positives"] > 0).all()


def test_assert_test_split_locked_enforced(monkeypatch):
    monkeypatch.setenv("ALLOW_TEST_SPLIT", "1")
    dummy_df = pd.DataFrame({"date": ["2023-01-01"], "y": [1], "s": [0.5]})
    with pytest.raises(RuntimeError, match="ALLOW_TEST_SPLIT"):
        run_block_bootstrap(
            df=dummy_df,
            label_col="y",
            score_cols={"climatology": "s", "G1": "s", "G2": "s", "G3": "s"},
        )
    monkeypatch.delenv("ALLOW_TEST_SPLIT", raising=False)

