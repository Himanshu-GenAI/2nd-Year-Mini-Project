"""Model v0 training and imbalance handling comparison (without weather).

Trains Logistic Regression and three XGBoost variants (plain, scale_pos_weight,
negative subsampling beta=0.2) using a strict two-stage early-stopping protocol
so 2023 validation labels never influence tree count selection.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT
from src.models.evaluate import brier, evaluate, record_result
from src.models.features_io import load_features
from src.utils.fit_registry import register_fit

V0_FEATURES = [
    "fire_count_7d",
    "fire_count_30d",
    "neighbour_fire_7d",
    "month_sin",
    "month_cos",
    "lat_c",
    "lon_c",
]
COUNT_FEATURES = ["fire_count_7d", "fire_count_30d", "neighbour_fire_7d"]

RUNS_CSV_PATH = PROJECT_ROOT / "docs" / "runs.csv"
COMPARISON_CSV_PATH = PROJECT_ROOT / "docs" / "audit" / "model_v0_comparison.csv"
RESULTS_CSV_PATH = PROJECT_ROOT / "docs" / "results.csv"


def prior_correct(p: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Apply analytical Bayes prior correction for negative subsampling at keep rate beta.

    Formula: p_corr = p / (p + (1 - p) / beta).
    Preserves boundary points 0 -> 0 and 1 -> 1; beta == 1 returns p unchanged.
    """
    p_arr = np.asarray(p, dtype=np.float64)
    if beta == 1.0:
        return float(p_arr) if np.ndim(p) == 0 else p_arr

    eps = 1e-15
    p_clipped = np.clip(p_arr, 0.0, 1.0)
    denom = p_clipped + (1.0 - p_clipped) / beta
    corrected = np.where(p_clipped <= 0.0, 0.0, np.where(p_clipped >= 1.0, 1.0, p_clipped / np.maximum(denom, eps)))
    if np.ndim(p) == 0:
        return float(corrected)
    return corrected


def subsample_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    beta: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Retain all positive rows and randomly keep `beta` fraction of negative rows."""
    rng = np.random.default_rng(seed)
    pos_mask = y.values == 1
    neg_idx = np.where(~pos_mask)[0]
    pos_idx = np.where(pos_mask)[0]

    n_keep_neg = int(round(beta * len(neg_idx)))
    sampled_neg_idx = rng.choice(neg_idx, size=n_keep_neg, replace=False)

    keep_idx = np.sort(np.concatenate([pos_idx, sampled_neg_idx]))
    return X.iloc[keep_idx].reset_index(drop=True), y.iloc[keep_idx].reset_index(drop=True)


def _prepare_logreg_matrix(X: pd.DataFrame) -> np.ndarray:
    X_copy = X.copy()
    for col in COUNT_FEATURES:
        X_copy[col] = np.log1p(np.maximum(X_copy[col].values, 0.0))
    return X_copy.values.astype(np.float32)


def fit_and_score_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series,
    X_val: pd.DataFrame,
    seed: int = 42,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """Fit log1p + StandardScaler + LogisticRegression on training data and score val."""
    first_d = pd.to_datetime(train_dates).min().date()
    last_d = pd.to_datetime(train_dates).max().date()

    X_tr_mat = _prepare_logreg_matrix(X_train)
    X_va_mat = _prepare_logreg_matrix(X_val)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_mat)
    X_va_scaled = scaler.transform(X_va_mat)

    register_fit(
        name="scaler_logreg_v0",
        first_date=first_d,
        last_date=last_d,
        metadata={"n_features": X_tr_mat.shape[1]},
    )

    clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=500, random_state=seed)
    clf.fit(X_tr_scaled, y_train.values)
    val_scores = clf.predict_proba(X_va_scaled)[:, 1]
    return clf, scaler, val_scores


def fit_and_score_xgb_two_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series,
    X_val: pd.DataFrame,
    variant: str = "plain",
    beta: float = 0.2,
    seed: int = 42,
) -> tuple[xgb.XGBClassifier, int, np.ndarray]:
    """Two-stage XGBoost fit: Stage 1 early-stops on 2022, Stage 2 refits 2015-2022."""
    dt = pd.to_datetime(train_dates)
    s1_train_mask = dt.dt.year <= 2021
    s1_es_mask = dt.dt.year == 2022

    X_s1_tr, y_s1_tr = X_train[s1_train_mask], y_train[s1_train_mask]
    X_s1_es, y_s1_es = X_train[s1_es_mask], y_train[s1_es_mask]

    if variant == "sub20":
        X_s1_tr_fit, y_s1_tr_fit = subsample_negatives(X_s1_tr, y_s1_tr, beta=beta, seed=seed)
    else:
        X_s1_tr_fit, y_s1_tr_fit = X_s1_tr, y_s1_tr

    spw_s1 = 1.0
    if variant == "spw":
        n_pos = max(int((y_s1_tr_fit == 1).sum()), 1)
        n_neg = int((y_s1_tr_fit == 0).sum())
        spw_s1 = n_neg / n_pos

    # Stage 1: find best iteration on 2022
    m1 = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=spw_s1,
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
    )
    m1.fit(
        X_s1_tr_fit,
        y_s1_tr_fit,
        eval_set=[(X_s1_es, y_s1_es)],
        verbose=False,
    )
    best_n_trees = int(m1.best_iteration + 1)

    # Stage 2: refit on complete 2015-2022 training window with fixed tree count
    if variant == "sub20":
        X_s2_fit, y_s2_fit = subsample_negatives(X_train, y_train, beta=beta, seed=seed)
    else:
        X_s2_fit, y_s2_fit = X_train, y_train

    spw_s2 = 1.0
    if variant == "spw":
        n_pos2 = max(int((y_s2_fit == 1).sum()), 1)
        n_neg2 = int((y_s2_fit == 0).sum())
        spw_s2 = n_neg2 / n_pos2

    m2 = xgb.XGBClassifier(
        n_estimators=best_n_trees,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=spw_s2,
        random_state=seed,
        n_jobs=-1,
    )
    m2.fit(X_s2_fit, y_s2_fit, verbose=False)
    val_scores = m2.predict_proba(X_val)[:, 1]
    return m2, best_n_trees, val_scores


def log_run(run_info: dict[str, Any], runs_path: Path = RUNS_CSV_PATH) -> None:
    """Append a model training run record to docs/runs.csv."""
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "date",
        "model",
        "feature_group",
        "label",
        "set_name",
        "parameters",
        "seconds",
        "pr_auc",
        "recall_top10",
        "precision_top5",
        "brier",
    ]
    write_header = not runs_path.exists()
    with open(runs_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(run_info)


def run_v0_experiment() -> pd.DataFrame:
    """Train and evaluate all Model v0 variants across both labels on 2023 val split."""
    all_comparisons: list[dict[str, Any]] = []

    # Include Week 2 Climatology baseline rows if available
    if RESULTS_CSV_PATH.exists():
        res_df = pd.read_csv(RESULTS_CSV_PATH)
        clim = res_df[res_df["model"] == "climatology"]
        for _, row in clim.iterrows():
            all_comparisons.append(
                {
                    "model": row["model"],
                    "label": row["label"],
                    "set_name": row["set_name"],
                    "base_rate": row["base_rate"],
                    "pr_auc": row["pr_auc"],
                    "pr_auc_over_base": row["pr_auc_over_base"],
                    "recall_top10": row["recall_top10"],
                    "precision_top5": row["precision_top5"],
                    "brier": row["brier"],
                    "brier_prior_corrected": row["brier"],
                    "best_trees": "-",
                    "seconds": 0.0,
                }
            )

    configs = [
        ("new_fire_next_48h", True, "primary"),
        ("fire_next_48h", False, "all"),
    ]

    run_counter = 1
    for label, primary, set_name in configs:
        print(f"\n=== Loading split data for {label} ({set_name}) ===")
        X_tr, y_tr, keys_tr = load_features("train", label=label, feature_names=V0_FEATURES, primary=primary)
        X_va, y_va, keys_va = load_features("val", label=label, feature_names=V0_FEATURES, primary=primary)

        # Prepare evaluation DataFrame
        eval_base = keys_va.copy()
        eval_base[label] = y_va.values

        # 1. Logistic Regression
        t0 = time.perf_counter()
        _, _, scores_lr = fit_and_score_logreg(X_tr, y_tr, keys_tr["date"], X_va, seed=42)
        dur = time.perf_counter() - t0
        eval_base["score"] = scores_lr
        res_lr = evaluate(eval_base, label, "score", "logreg_v0", "val", set_name)
        record_result(res_lr)
        log_run(
            {
                "run_id": f"v0_{run_counter:02d}",
                "date": str(date.today()),
                "model": "logreg_v0",
                "feature_group": "G1_v0",
                "label": label,
                "set_name": set_name,
                "parameters": json.dumps({"C": 1.0, "max_iter": 500, "log1p_counts": True}),
                "seconds": round(dur, 2),
                "pr_auc": res_lr["pr_auc"],
                "recall_top10": res_lr["recall_top10"],
                "precision_top5": res_lr["precision_top5"],
                "brier": res_lr["brier"],
            }
        )
        run_counter += 1
        all_comparisons.append(
            {
                "model": "logreg_v0",
                "label": label,
                "set_name": set_name,
                "base_rate": res_lr["base_rate"],
                "pr_auc": res_lr["pr_auc"],
                "pr_auc_over_base": res_lr["pr_auc_over_base"],
                "recall_top10": res_lr["recall_top10"],
                "precision_top5": res_lr["precision_top5"],
                "brier": res_lr["brier"],
                "brier_prior_corrected": res_lr["brier"],
                "best_trees": "-",
                "seconds": round(dur, 2),
            }
        )
        print(f"  logreg_v0: PR-AUC={res_lr['pr_auc']:.4f}, Recall@10%={res_lr['recall_top10']:.4f}, Brier={res_lr['brier']:.6f} ({dur:.1f}s)")

        # 2-4. XGBoost variants
        for variant, m_name in [
            ("plain", "xgb_v0_plain"),
            ("spw", "xgb_v0_spw"),
            ("sub20", "xgb_v0_sub20"),
        ]:
            t0 = time.perf_counter()
            _, n_trees, scores_xgb = fit_and_score_xgb_two_stage(
                X_tr, y_tr, keys_tr["date"], X_va, variant=variant, beta=0.2, seed=42
            )
            dur = time.perf_counter() - t0
            eval_base["score"] = scores_xgb
            res_xgb = evaluate(eval_base, label, "score", m_name, "val", set_name)
            record_result(res_xgb)

            brier_corr = res_xgb["brier"]
            if variant == "sub20":
                eval_base["score_corr"] = prior_correct(scores_xgb, beta=0.2)
                brier_corr = round(brier(eval_base, label, "score_corr"), 6)

            log_run(
                {
                    "run_id": f"v0_{run_counter:02d}",
                    "date": str(date.today()),
                    "model": m_name,
                    "feature_group": "G1_v0",
                    "label": label,
                    "set_name": set_name,
                    "parameters": json.dumps(
                        {"variant": variant, "best_trees": n_trees, "max_depth": 6, "lr": 0.1, "beta": 0.2 if variant == "sub20" else 1.0}
                    ),
                    "seconds": round(dur, 2),
                    "pr_auc": res_xgb["pr_auc"],
                    "recall_top10": res_xgb["recall_top10"],
                    "precision_top5": res_xgb["precision_top5"],
                    "brier": res_xgb["brier"],
                }
            )
            run_counter += 1
            all_comparisons.append(
                {
                    "model": m_name,
                    "label": label,
                    "set_name": set_name,
                    "base_rate": res_xgb["base_rate"],
                    "pr_auc": res_xgb["pr_auc"],
                    "pr_auc_over_base": res_xgb["pr_auc_over_base"],
                    "recall_top10": res_xgb["recall_top10"],
                    "precision_top5": res_xgb["precision_top5"],
                    "brier": res_xgb["brier"],
                    "brier_prior_corrected": brier_corr,
                    "best_trees": n_trees,
                    "seconds": round(dur, 2),
                }
            )
            print(
                f"  {m_name} (trees={n_trees}): PR-AUC={res_xgb['pr_auc']:.4f}, "
                f"Recall@10%={res_xgb['recall_top10']:.4f}, Brier={res_xgb['brier']:.6f}, "
                f"Brier_corr={brier_corr:.6f} ({dur:.1f}s)"
            )

    comp_df = pd.DataFrame(all_comparisons)
    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\nComparison table saved to {COMPARISON_CSV_PATH}")
    return comp_df


if __name__ == "__main__":
    run_v0_experiment()

