"""Diagnostic evaluations on partial cached weather cells (Day 5 / Week 3).

Provides:
1. Block bootstrap by calendar week of 2023 for PR-AUC differences (G2-G1, G3-G2, G3-climatology).
2. Rolling-origin evaluation for validation years 2021 and 2022 (with test split strictly locked).
3. KBDI ablation check (G1K = G1 + KBDI; G2R = G2 - rainfall sums/days_since_rain + KBDI).
4. Fire-season evaluation on peak positive fire months determined from 2015-2022 training data.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import FIRE_TABLE_PATH
from src.features.build_features import ALL_FEATURE_NAMES
from src.models.baselines import fit_climatology, predict_climatology
from src.models.evaluate import brier, evaluate, precision_at_top_frac, recall_at_top_frac
from src.models.features_io import load_features
from src.models.train_v1 import (
    DEFAULT_XGB_PARAMS,
    FEATURE_GROUPS,
    FEATURES_PARTIAL_PATH,
    assert_test_split_locked,
    fit_xgb_two_stage,
)

# KBDI diagnostic feature groups
KBDI_FEATURES = ["kbdi", "kbdi_change_7d"]
RAINFALL_ACCUM_FEATURES = ["precip_sum_7d", "precip_sum_30d", "days_since_rain"]

DIAGNOSTIC_FEATURE_GROUPS: dict[str, list[str]] = {
    "G1": list(FEATURE_GROUPS["G1"]),
    "G1K": [*FEATURE_GROUPS["G1"], *KBDI_FEATURES],
    "G2": list(FEATURE_GROUPS["G2"]),
    "G2R": [f for f in FEATURE_GROUPS["G3"] if f not in RAINFALL_ACCUM_FEATURES],
    "G3": list(FEATURE_GROUPS["G3"]),
}


def get_calendar_weeks(dates: pd.Series) -> pd.Series:
    """Map dates to weekly period strings (calendar weeks ending Sunday)."""
    return pd.to_datetime(dates).dt.to_period("W").astype(str)


def run_block_bootstrap(
    df: pd.DataFrame,
    label_col: str,
    score_cols: dict[str, str],
    date_col: str = "date",
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run block bootstrap by calendar week of 2023 for model PR-AUC differences.

    Args:
        df: Validation DataFrame with labels and model scores.
        label_col: Target column name.
        score_cols: Mapping of model keys ('climatology', 'G1', 'G2', 'G3') to column names.
        date_col: Date column name.
        n_boot: Number of bootstrap iterations (default 1,000).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (replicates_df, summary_df).
    """
    assert_test_split_locked()
    y_true = df[label_col].values.astype(int)
    s_clim = df[score_cols["climatology"]].values.astype(float)
    s_g1 = df[score_cols["G1"]].values.astype(float)
    s_g2 = df[score_cols["G2"]].values.astype(float)
    s_g3 = df[score_cols["G3"]].values.astype(float)

    weeks = get_calendar_weeks(df[date_col]).values
    unique_weeks = np.unique(weeks)
    n_unique_weeks = len(unique_weeks)

    # Pre-index row indices by week for performance
    week_indices = {w: np.where(weeks == w)[0] for w in unique_weeks}

    rng = np.random.default_rng(seed)

    replicates = []
    for b in range(n_boot):
        sampled_weeks = rng.choice(unique_weeks, size=n_unique_weeks, replace=True)
        idx_b = np.concatenate([week_indices[w] for w in sampled_weeks])

        y_b = y_true[idx_b]
        if y_b.sum() == 0:
            continue

        pr_clim = float(average_precision_score(y_b, s_clim[idx_b]))
        pr_g1 = float(average_precision_score(y_b, s_g1[idx_b]))
        pr_g2 = float(average_precision_score(y_b, s_g2[idx_b]))
        pr_g3 = float(average_precision_score(y_b, s_g3[idx_b]))

        replicates.append(
            {
                "replicate": b + 1,
                "diff_G2_G1": pr_g2 - pr_g1,
                "diff_G3_G2": pr_g3 - pr_g2,
                "diff_G3_clim": pr_g3 - pr_clim,
                "pr_clim": pr_clim,
                "pr_G1": pr_g1,
                "pr_G2": pr_g2,
                "pr_G3": pr_g3,
            }
        )

    draws_df = pd.DataFrame(replicates)

    # Compute empirical percentiles (95% CI: 2.5th to 97.5th percentile)
    contrasts = [
        ("G2 - G1", "diff_G2_G1"),
        ("G3 - G2", "diff_G3_G2"),
        ("G3 - climatology", "diff_G3_clim"),
    ]

    base_pr_clim = float(average_precision_score(y_true, s_clim))
    base_pr_g1 = float(average_precision_score(y_true, s_g1))
    base_pr_g2 = float(average_precision_score(y_true, s_g2))
    base_pr_g3 = float(average_precision_score(y_true, s_g3))
    point_diffs = {
        "diff_G2_G1": base_pr_g2 - base_pr_g1,
        "diff_G3_G2": base_pr_g3 - base_pr_g2,
        "diff_G3_clim": base_pr_g3 - base_pr_clim,
    }

    summary_rows = []
    for contrast_name, col in contrasts:
        vals = draws_df[col].values
        ci_low = float(np.percentile(vals, 2.5))
        ci_high = float(np.percentile(vals, 97.5))
        inc_zero = bool(ci_low <= 0.0 <= ci_high)
        summary_rows.append(
            {
                "contrast": contrast_name,
                "point_diff": round(point_diffs[col], 6),
                "boot_mean_diff": round(float(np.mean(vals)), 6),
                "boot_std_err": round(float(np.std(vals, ddof=1)), 6),
                "ci_lower_95": round(ci_low, 6),
                "ci_upper_95": round(ci_high, 6),
                "includes_zero": inc_zero,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return draws_df, summary_df


def run_rolling_origin_eval(
    val_year: int,
    partial_cells: list[str],
    features_path: Path | None = None,
    fire_table_path: Path = FIRE_TABLE_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    """Run rolling-origin evaluation for validation year 2021 or 2022.

    For val_year == 2022: train 2015-2020, early-stop 2021, refit 2015-2021, score 2022.
    For val_year == 2021: train 2015-2019, early-stop 2020, refit 2015-2020, score 2021.
    Strictly forbids 2024-2025.
    """
    assert_test_split_locked()
    if val_year not in (2021, 2022):
        raise ValueError(f"Rolling origin supports val_year in (2021, 2022); got {val_year}")

    s1_es_year = val_year - 1
    s1_train_end_year = s1_es_year - 1

    f_path = features_path or FEATURES_PARTIAL_PATH
    label = "new_fire_next_48h"
    n_cells = len(partial_cells)
    set_name = f"partial: {n_cells} cells"

    # Step 1: Load all training split rows (spans 2015 to 2022)
    X_all_train, y_all_train, keys_all_train = load_features(
        "train",
        label=label,
        feature_names=ALL_FEATURE_NAMES,
        primary=True,
        features_path=f_path,
        fire_table_path=fire_table_path,
        allowed_cells=partial_cells,
    )

    dates = pd.to_datetime(keys_all_train["date"])
    train_mask = dates.dt.year <= s1_es_year
    val_mask = dates.dt.year == val_year

    X_tr = X_all_train[train_mask].copy()
    y_tr = y_all_train[train_mask].copy()
    keys_tr = keys_all_train[train_mask].copy()

    X_va = X_all_train[val_mask].copy()
    y_va = y_all_train[val_mask].copy()
    keys_va = keys_all_train[val_mask].copy()

    # Step 2: Fit and score climatology
    clim_train_df = keys_tr[["cell_id", "date"]].copy()
    clim_train_df[label] = y_tr.values
    clim_train_df["burning_prev3"] = False

    eval_clim = keys_va[["cell_id", "date", "burning_prev3", "split"]].copy()
    eval_clim[label] = y_va.values

    clim_model = fit_climatology(clim_train_df, label=label, alpha=1.0)
    clim_scores = predict_climatology(clim_model, eval_clim)
    eval_clim["score"] = clim_scores
    res_clim = evaluate(
        eval_clim,
        label,
        "score",
        model="climatology",
        split=f"val_{val_year}",
        set_name=set_name,
    )
    res_clim["recall_top10"] = round(float(recall_at_top_frac(eval_clim, label, "score", frac=0.10)[0]), 4)
    res_clim["set_name"] = set_name

    results_rows = [
        {
            "val_year": val_year,
            "group": "climatology",
            "n_features": 0,
            "label": label,
            "set_name": set_name,
            "base_rate": res_clim["base_rate"],
            "pr_auc": res_clim["pr_auc"],
            "pr_auc_over_base": res_clim["pr_auc_over_base"],
            "recall_top10": res_clim["recall_top10"],
            "precision_top5": res_clim["precision_top5"],
            "brier": res_clim["brier"],
            "best_trees": 0,
            "seconds": 0.0,
        }
    ]

    # Step 3: Fit and score G1, G2, G3
    for group_name in ["G1", "G2", "G3"]:
        cols = FEATURE_GROUPS[group_name]
        t0 = time.perf_counter()
        _, best_trees, scores = fit_xgb_two_stage(
            X_tr[cols],
            y_tr,
            keys_tr["date"],
            X_va[cols],
            params=DEFAULT_XGB_PARAMS,
            max_trees=600,
            early_stopping_rounds=30,
            seed=seed,
            s1_train_end_year=s1_train_end_year,
            s1_es_year=s1_es_year,
        )
        dur = time.perf_counter() - t0

        eval_df = keys_va.copy()
        eval_df[label] = y_va.values
        eval_df["score"] = scores
        res = evaluate(
            eval_df,
            label,
            "score",
            model=f"xgb_v1_{group_name}",
            split=f"val_{val_year}",
            set_name=set_name,
        )
        res["recall_top10"] = round(float(recall_at_top_frac(eval_df, label, "score", frac=0.10)[0]), 4)
        res["set_name"] = set_name

        results_rows.append(
            {
                "val_year": val_year,
                "group": group_name,
                "n_features": len(cols),
                "label": label,
                "set_name": set_name,
                "base_rate": res["base_rate"],
                "pr_auc": res["pr_auc"],
                "pr_auc_over_base": res["pr_auc_over_base"],
                "recall_top10": res["recall_top10"],
                "precision_top5": res["precision_top5"],
                "brier": res["brier"],
                "best_trees": best_trees,
                "seconds": round(dur, 2),
            }
        )

    return pd.DataFrame(results_rows)


def run_kbdi_ablation(
    partial_cells: list[str],
    features_path: Path | None = None,
    fire_table_path: Path = FIRE_TABLE_PATH,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate G1, G1K, G2, G2R, G3 on 2023 validation primary set.

    Returns:
        Tuple of (results_table, val_predictions_df).
    """
    assert_test_split_locked()
    f_path = features_path or FEATURES_PARTIAL_PATH
    label = "new_fire_next_48h"
    n_cells = len(partial_cells)
    set_name = f"partial: {n_cells} cells"

    X_tr, y_tr, keys_tr = load_features(
        "train",
        label=label,
        feature_names=ALL_FEATURE_NAMES,
        primary=True,
        features_path=f_path,
        fire_table_path=fire_table_path,
        allowed_cells=partial_cells,
    )
    X_va, y_va, keys_va = load_features(
        "val",
        label=label,
        feature_names=ALL_FEATURE_NAMES,
        primary=True,
        features_path=f_path,
        fire_table_path=fire_table_path,
        allowed_cells=partial_cells,
    )

    preds_df = keys_va[["cell_id", "date"]].copy()
    preds_df[label] = y_va.values

    # Fit and score climatology for comparison
    raw_fire_df = pd.read_parquet(fire_table_path)
    raw_fire_df = raw_fire_df[raw_fire_df["cell_id"].astype(str).isin(set(partial_cells))].copy()
    clim_train = raw_fire_df[raw_fire_df["split"] == "train"].copy()
    clim_val = raw_fire_df[raw_fire_df["split"] == "val"].copy()
    clim_val = clim_val[~clim_val["burning_prev3"].astype(bool)].copy()
    clim_model = fit_climatology(clim_train, label=label, alpha=1.0)
    preds_df["score_climatology"] = predict_climatology(clim_model, clim_val)

    rows = []
    for g_name in ["G1", "G1K", "G2", "G2R", "G3"]:
        cols = DIAGNOSTIC_FEATURE_GROUPS[g_name]
        t0 = time.perf_counter()
        _, best_trees, scores = fit_xgb_two_stage(
            X_tr[cols],
            y_tr,
            keys_tr["date"],
            X_va[cols],
            params=DEFAULT_XGB_PARAMS,
            max_trees=600,
            early_stopping_rounds=30,
            seed=seed,
        )
        dur = time.perf_counter() - t0
        preds_df[f"score_{g_name}"] = scores

        eval_df = keys_va.copy()
        eval_df[label] = y_va.values
        eval_df["score"] = scores
        res = evaluate(
            eval_df,
            label,
            "score",
            model=f"xgb_v1_{g_name}",
            split="val",
            set_name=set_name,
        )
        res["recall_top10"] = round(float(recall_at_top_frac(eval_df, label, "score", frac=0.10)[0]), 4)
        res["set_name"] = set_name

        rows.append(
            {
                "group": g_name,
                "n_features": len(cols),
                "label": label,
                "set_name": set_name,
                "base_rate": res["base_rate"],
                "pr_auc": res["pr_auc"],
                "pr_auc_over_base": res["pr_auc_over_base"],
                "recall_top10": res["recall_top10"],
                "precision_top5": res["precision_top5"],
                "brier": res["brier"],
                "best_trees": best_trees,
                "seconds": round(dur, 2),
            }
        )

    return pd.DataFrame(rows), preds_df


def evaluate_fire_season(
    preds_df: pd.DataFrame,
    peak_months: list[int],
    label: str = "new_fire_next_48h",
    set_name: str = "partial: 346 cells",
) -> pd.DataFrame:
    """Evaluate model predictions restricted to specified calendar months."""
    months = pd.to_datetime(preds_df["date"]).dt.month
    season_mask = months.isin(peak_months)
    season_df = preds_df[season_mask].copy()

    month_str = "-".join(str(m) for m in sorted(peak_months))
    models = [
        ("climatology", "score_climatology", 0),
        ("G1", "score_G1", 7),
        ("G2", "score_G2", 19),
        ("G3", "score_G3", 21),
    ]

    rows = []
    n_rows = len(season_df)
    n_pos = int(season_df[label].sum())
    base_r = float(season_df[label].mean())

    for name, col, n_feats in models:
        prauc = float(average_precision_score(season_df[label].values, season_df[col].values))
        prauc_over_base = prauc / base_r if base_r > 0 else float("nan")
        rec10, _ = recall_at_top_frac(season_df, label, col, frac=0.10)
        prec5 = precision_at_top_frac(season_df, label, col, frac=0.05)
        br = brier(season_df, label, col)

        rows.append(
            {
                "season_months": month_str,
                "group": name,
                "n_features": n_feats,
                "label": label,
                "set_name": set_name,
                "rows": n_rows,
                "positives": n_pos,
                "base_rate": round(base_r, 6),
                "pr_auc": round(prauc, 6),
                "pr_auc_over_base": round(prauc_over_base, 3),
                "recall_top10": round(rec10, 4),
                "precision_top5": round(prec5, 4),
                "brier": round(br, 6),
            }
        )

    return pd.DataFrame(rows)
