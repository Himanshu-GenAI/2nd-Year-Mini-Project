"""Model v1 training, ablation, hyperparameter tuning, Random Forest benchmark, and feature importance.

Supports full (1,167 cells) and partial (>=300 cached cells) modes, enforces strict
test-split lock (`ALLOW_TEST_SPLIT` must remain unset), and prevents 2023 validation
leakage during XGBoost tree-count selection via two-stage training (Stage 1: 2015-2021
fit / 2022 early stop; Stage 2: 2015-2022 refit).
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from src.config import DOCS_AUDIT_DIR, DOCS_DIR, FIRE_TABLE_PATH, PROCESSED_DIR, PROJECT_ROOT
from src.features.build_features import ALL_FEATURE_NAMES, BASE_FIRE_FEATURE_NAMES
from src.models.baselines import fit_climatology, predict_climatology
from src.models.evaluate import evaluate, recall_at_top_frac, record_result
from src.models.features_io import FEATURES_FULL_PATH, load_features
from src.models.train_v0 import V0_FEATURES, fit_and_score_xgb_two_stage, log_run

FEATURES_PARTIAL_PATH = PROCESSED_DIR / "features_partial.parquet"
PARTIAL_CELLS_PATH = PROCESSED_DIR / "partial_cells.json"
ABLATION_CSV_PATH = DOCS_AUDIT_DIR / "ablation.csv"
FEATURE_IMPORTANCE_CSV_PATH = DOCS_AUDIT_DIR / "feature_importance.csv"
RUNS_CSV_PATH = DOCS_DIR / "runs.csv"
RESULTS_CSV_PATH = DOCS_DIR / "results.csv"
RESULTS_MD_PATH = DOCS_DIR / "results.md"
TUNED_XGB_MODEL_DIR = PROJECT_ROOT / "models" / "xgb_v1_tuned"

DROUGHT_FEATURES = ["kbdi", "kbdi_change_7d"]
WEATHER_LAG_FEATURES = [f for f in ALL_FEATURE_NAMES if f not in BASE_FIRE_FEATURE_NAMES and f not in DROUGHT_FEATURES]

FEATURE_GROUPS: dict[str, list[str]] = {
    "G1": list(BASE_FIRE_FEATURE_NAMES),
    "G2": [*BASE_FIRE_FEATURE_NAMES, *WEATHER_LAG_FEATURES],
    "G3": list(ALL_FEATURE_NAMES),
}

TUNING_SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 20.0],
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def assert_test_split_locked() -> None:
    """Fail loudly if ALLOW_TEST_SPLIT is set to any truthy/non-empty value."""
    val = os.environ.get("ALLOW_TEST_SPLIT", "").strip()
    if val:
        raise RuntimeError(f"STOP: ALLOW_TEST_SPLIT is set ({val!r}); 2024-2025 test split must stay locked.")


def _format_set_name(base_set: str, partial: bool, n_cells: int | None) -> str:
    if partial and n_cells is not None:
        return f"partial: {n_cells} cells" if base_set == "primary" else f"partial: {n_cells} cells ({base_set})"
    return base_set


def load_partial_cells(partial_cells_path: Path = PARTIAL_CELLS_PATH) -> list[str]:
    """Read cached cell IDs list saved during partial feature build."""
    if not partial_cells_path.exists():
        raise FileNotFoundError(f"Partial cell list not found: {partial_cells_path}")
    data = json.loads(partial_cells_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cells" in data:
        return [str(c) for c in data["cells"]]
    if isinstance(data, list):
        return [str(c) for c in data]
    raise ValueError(f"Unrecognized partial_cells.json format in {partial_cells_path}")


def fit_xgb_two_stage(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series,
    X_val: pd.DataFrame,
    params: dict[str, Any] | None = None,
    max_trees: int = 1000,
    early_stopping_rounds: int = 30,
    seed: int = 42,
    n_jobs: int = -1,
) -> tuple[xgb.XGBClassifier, int, np.ndarray]:
    """Two-stage XGBoost fit: Stage 1 early-stops on 2022, Stage 2 refits 2015-2022."""
    dt = pd.to_datetime(train_dates)
    if int(dt.dt.year.max()) > 2022:
        raise ValueError(f"Training dates must be <= 2022; got max year {int(dt.dt.year.max())}")

    s1_train_mask = dt.dt.year <= 2021
    s1_es_mask = dt.dt.year == 2022
    if int(s1_train_mask.sum()) == 0 or int(s1_es_mask.sum()) == 0:
        raise ValueError("Two-stage XGBoost requires non-empty <=2021 train and ==2022 early-stopping rows.")

    cfg = {**DEFAULT_XGB_PARAMS, **(params or {})}
    X_s1_tr, y_s1_tr = X_train[s1_train_mask], y_train[s1_train_mask]
    X_s1_es, y_s1_es = X_train[s1_es_mask], y_train[s1_es_mask]

    m1 = xgb.XGBClassifier(
        n_estimators=max_trees,
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        min_child_weight=float(cfg["min_child_weight"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_lambda=float(cfg["reg_lambda"]),
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=1.0,
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=n_jobs,
    )
    m1.fit(X_s1_tr, y_s1_tr, eval_set=[(X_s1_es, y_s1_es)], verbose=False)
    best_n_trees = int(m1.best_iteration + 1) if m1.best_iteration is not None else max_trees

    m2 = xgb.XGBClassifier(
        n_estimators=best_n_trees,
        max_depth=int(cfg["max_depth"]),
        learning_rate=float(cfg["learning_rate"]),
        min_child_weight=float(cfg["min_child_weight"]),
        subsample=float(cfg["subsample"]),
        colsample_bytree=float(cfg["colsample_bytree"]),
        reg_lambda=float(cfg["reg_lambda"]),
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=1.0,
        random_state=seed,
        n_jobs=n_jobs,
    )
    m2.fit(X_train, y_train, verbose=False)
    val_scores = m2.predict_proba(X_val)[:, 1]
    return m2, best_n_trees, val_scores


def sample_tuning_configs(n_trials: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """Sample unique random-search configurations deterministically."""
    rng = np.random.default_rng(seed)
    keys = list(TUNING_SEARCH_SPACE.keys())
    seen: set[tuple[Any, ...]] = set()
    configs: list[dict[str, Any]] = []
    max_attempts = n_trials * 50
    attempts = 0
    while len(configs) < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = {}
        sig = []
        for k in keys:
            choices = TUNING_SEARCH_SPACE[k]
            val = choices[int(rng.integers(0, len(choices)))]
            cfg[k] = val
            sig.append(val)
        tup = tuple(sig)
        if tup not in seen:
            seen.add(tup)
            configs.append(cfg)
    return configs


def select_winner(candidates: list[dict[str, Any]], pr_tol: float = 0.005) -> dict[str, Any]:
    """Select winning model: highest PR-AUC; if within 0.005 use Recall@10%; tie-break simpler."""
    if not candidates:
        raise ValueError("Cannot select winner from empty candidate list.")

    max_pr = max(float(c["pr_auc"]) for c in candidates)
    near_best = [c for c in candidates if float(c["pr_auc"]) >= max_pr - pr_tol]
    # Primary order: highest recall_top10, then lowest complexity, then highest pr_auc, then name
    ranked = sorted(
        near_best,
        key=lambda c: (
            -round(float(c["recall_top10"]), 6),
            float(c.get("complexity", 999)),
            -float(c["pr_auc"]),
            str(c.get("model", "")),
        ),
    )
    return ranked[0]


def compute_partial_climatology(
    partial_cells: list[str],
    fire_table_path: Path = FIRE_TABLE_PATH,
    results_csv_path: Path = RESULTS_CSV_PATH,
    results_md_path: Path = RESULTS_MD_PATH,
    primary_only: bool = False,
) -> list[dict[str, Any]]:
    """Recompute climatology baseline on identical partial cells for fair comparison."""
    assert_test_split_locked()
    df = pd.read_parquet(fire_table_path)
    df = df[df["cell_id"].astype(str).isin(set(partial_cells)) & df["split"].isin(["train", "val"])].copy()
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    n_cells = len(partial_cells)
    targets = [("new_fire_next_48h", "primary")] if primary_only else [("new_fire_next_48h", "primary"), ("fire_next_48h", "all")]
    rows: list[dict[str, Any]] = []
    for label, base_set in targets:
        clim = fit_climatology(train_df, label=label, alpha=1.0)
        eval_df = val_df.copy()
        eval_df["score"] = predict_climatology(clim, eval_df)
        set_name = _format_set_name(base_set, partial=True, n_cells=n_cells)
        res = evaluate(
            eval_df if base_set == "all" else eval_df[~eval_df["burning_prev3"].astype(bool)],
            label,
            "score",
            "climatology",
            "val",
            set_name=set_name,
        )
        record_result(res, csv_path=results_csv_path, md_path=results_md_path)
        rows.append(res)
    return rows


def run_ablation(
    partial: bool = False,
    features_path: Path | None = None,
    fire_table_path: Path = FIRE_TABLE_PATH,
    partial_cells_path: Path = PARTIAL_CELLS_PATH,
    ablation_csv_path: Path = ABLATION_CSV_PATH,
    runs_csv_path: Path = RUNS_CSV_PATH,
    results_csv_path: Path = RESULTS_CSV_PATH,
    results_md_path: Path = RESULTS_MD_PATH,
    primary_only: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Run G1, G2, G3 ablation (plus identical-cell baselines when partial) using plain XGBoost."""
    assert_test_split_locked()
    f_path = features_path or (FEATURES_PARTIAL_PATH if partial else FEATURES_FULL_PATH)

    ablation_rows: list[dict[str, Any]] = []
    clim_primary_pr: float | None = None
    n_cells: int | None = None
    partial_cells: list[str] | None = None
    if partial:
        partial_cells = load_partial_cells(partial_cells_path)
        n_cells = len(partial_cells)
        clim_rows = compute_partial_climatology(
            partial_cells=partial_cells,
            fire_table_path=fire_table_path,
            results_csv_path=results_csv_path,
            results_md_path=results_md_path,
            primary_only=primary_only,
        )
        for r in clim_rows:
            if r["label"] == "new_fire_next_48h":
                clim_primary_pr = float(r["pr_auc"])
            ablation_rows.append(
                {
                    "group": "climatology",
                    "n_features": 0,
                    "label": r["label"],
                    "set_name": r["set_name"],
                    "base_rate": r["base_rate"],
                    "pr_auc": r["pr_auc"],
                    "pr_auc_over_base": r["pr_auc_over_base"],
                    "recall_top10": r["recall_top10"],
                    "precision_top5": r["precision_top5"],
                    "brier": r["brier"],
                    "best_trees": 0,
                    "seconds": 0.0,
                }
            )

    run_idx = 1

    label_configs = (
        [("new_fire_next_48h", True, "primary")]
        if primary_only
        else [("new_fire_next_48h", True, "primary"), ("fire_next_48h", False, "all")]
    )

    for label, primary, base_set in label_configs:
        set_name = _format_set_name(base_set, partial=partial, n_cells=n_cells)
        X_tr_all, y_tr, keys_tr = load_features(
            "train",
            label=label,
            feature_names=ALL_FEATURE_NAMES,
            primary=primary,
            features_path=f_path,
            fire_table_path=fire_table_path,
            allowed_cells=partial_cells,
        )
        X_va_all, y_va, keys_va = load_features(
            "val",
            label=label,
            feature_names=ALL_FEATURE_NAMES,
            primary=primary,
            features_path=f_path,
            fire_table_path=fire_table_path,
            allowed_cells=partial_cells,
        )

        # Recompute Model v0 (plain XGBoost) on the exact same cells when partial=True
        if partial:
            t0_v0 = time.perf_counter()
            _, v0_trees, v0_scores = fit_and_score_xgb_two_stage(
                X_tr_all[V0_FEATURES],
                y_tr,
                keys_tr["date"],
                X_va_all[V0_FEATURES],
                variant="plain",
                seed=seed,
            )
            dur_v0 = time.perf_counter() - t0_v0
            eval_v0 = keys_va.copy()
            eval_v0[label] = y_va.values
            eval_v0["score"] = v0_scores
            res_v0 = evaluate(
                eval_v0,
                label,
                "score",
                model="xgb_v0_plain",
                split="val",
                set_name=set_name,
            )
            if base_set == "primary":
                res_v0["recall_top10"] = round(float(recall_at_top_frac(eval_v0, label, "score", frac=0.10)[0]), 4)
            res_v0["set_name"] = set_name
            record_result(res_v0, csv_path=results_csv_path, md_path=results_md_path)
            log_run(
                {
                    "run_id": f"v0_part_{run_idx:02d}",
                    "date": str(date.today()),
                    "model": "xgb_v0_plain",
                    "feature_group": f"V0 ({len(V0_FEATURES)} cols)",
                    "label": label,
                    "set_name": set_name,
                    "parameters": json.dumps({"variant": "plain", "best_trees": v0_trees, "max_depth": 6, "lr": 0.1}),
                    "seconds": round(dur_v0, 2),
                    "pr_auc": res_v0["pr_auc"],
                    "recall_top10": res_v0["recall_top10"],
                    "precision_top5": res_v0["precision_top5"],
                    "brier": res_v0["brier"],
                },
                runs_path=runs_csv_path,
            )
            ablation_rows.append(
                {
                    "group": "xgb_v0_plain",
                    "n_features": len(V0_FEATURES),
                    "label": label,
                    "set_name": set_name,
                    "base_rate": res_v0["base_rate"],
                    "pr_auc": res_v0["pr_auc"],
                    "pr_auc_over_base": res_v0["pr_auc_over_base"],
                    "recall_top10": res_v0["recall_top10"],
                    "precision_top5": res_v0["precision_top5"],
                    "brier": res_v0["brier"],
                    "best_trees": v0_trees,
                    "seconds": round(dur_v0, 2),
                }
            )

        for group_name in ["G1", "G2", "G3"]:
            cols = FEATURE_GROUPS[group_name]
            t0 = time.perf_counter()
            _, best_trees, val_scores = fit_xgb_two_stage(
                X_tr_all[cols],
                y_tr,
                keys_tr["date"],
                X_va_all[cols],
                params=DEFAULT_XGB_PARAMS,
                max_trees=600,
                seed=seed,
            )
            dur = time.perf_counter() - t0

            eval_df = keys_va.copy()
            eval_df[label] = y_va.values
            eval_df["score"] = val_scores
            # Evaluate on pre-filtered primary/all DataFrame directly
            res = evaluate(
                eval_df,
                label,
                "score",
                model=f"xgb_v1_{group_name}",
                split="val",
                set_name=base_set if not partial else set_name,
            )
            # Keep exact partial tag even when set_name != 'primary'
            if base_set == "primary":
                res["recall_top10"] = round(float(recall_at_top_frac(eval_df, label, "score", frac=0.10)[0]), 4)
            res["set_name"] = set_name
            record_result(res, csv_path=results_csv_path, md_path=results_md_path)

            log_run(
                {
                    "run_id": f"v1_abl_{run_idx:02d}",
                    "date": str(date.today()),
                    "model": f"xgb_v1_{group_name}",
                    "feature_group": f"{group_name} ({len(cols)} cols)",
                    "label": label,
                    "set_name": set_name,
                    "parameters": json.dumps({**DEFAULT_XGB_PARAMS, "best_trees": best_trees}),
                    "seconds": round(dur, 2),
                    "pr_auc": res["pr_auc"],
                    "recall_top10": res["recall_top10"],
                    "precision_top5": res["precision_top5"],
                    "brier": res["brier"],
                },
                runs_path=runs_csv_path,
            )
            run_idx += 1

            ablation_rows.append(
                {
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

            if partial and label == "new_fire_next_48h" and group_name == "G3" and clim_primary_pr is not None:
                if float(res["pr_auc"]) < clim_primary_pr:
                    raise RuntimeError(
                        f"STOP: G3 PR-AUC ({res['pr_auc']:.6f}) is below climatology ({clim_primary_pr:.6f}) "
                        f"on partial {n_cells} cells."
                    )

    df_out = pd.DataFrame(ablation_rows)
    ablation_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(ablation_csv_path, index=False)
    return df_out


def run_xgb_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_dates: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_keys: pd.DataFrame,
    feature_group_name: str = "G3",
    n_trials: int = 20,
    max_minutes: float = 60.0,
    max_trees: int = 1000,
    partial: bool = False,
    n_cells: int | None = None,
    seed: int = 42,
    runs_csv_path: Path = RUNS_CSV_PATH,
    results_csv_path: Path = RESULTS_CSV_PATH,
    results_md_path: Path = RESULTS_MD_PATH,
    model_save_path: Path | None = None,
) -> tuple[xgb.XGBClassifier, dict[str, Any], pd.DataFrame, np.ndarray]:
    """Run deterministic random search for XGBoost (two-stage fit), obeying wall-clock cap."""
    assert_test_split_locked()
    configs = sample_tuning_configs(n_trials=n_trials, seed=seed)
    set_name = _format_set_name("primary", partial=partial, n_cells=n_cells)

    trial_records: list[dict[str, Any]] = []
    best_model: xgb.XGBClassifier | None = None
    best_cfg: dict[str, Any] | None = None
    best_scores: np.ndarray | None = None
    best_pr = -1.0
    best_trees_win = 0
    t_start = time.perf_counter()

    eval_df = val_keys.copy()
    eval_df["new_fire_next_48h"] = y_val.values

    for idx, cfg in enumerate(configs, start=1):
        elapsed_min = (time.perf_counter() - t_start) / 60.0
        if idx > 1 and elapsed_min >= max_minutes:
            break

        t0 = time.perf_counter()
        model, best_trees, scores = fit_xgb_two_stage(
            X_train=X_train,
            y_train=y_train,
            train_dates=train_dates,
            X_val=X_val,
            params=cfg,
            max_trees=max_trees,
            seed=seed,
        )
        dur = time.perf_counter() - t0
        eval_df["score"] = scores
        trial_res = evaluate(eval_df, "new_fire_next_48h", "score", model=f"xgb_tune_t{idx:02d}", split="val", set_name="primary")

        log_run(
            {
                "run_id": f"v1_tune_{idx:02d}",
                "date": str(date.today()),
                "model": f"xgb_v1_tune_{idx:02d}",
                "feature_group": feature_group_name,
                "label": "new_fire_next_48h",
                "set_name": set_name,
                "parameters": json.dumps({**cfg, "best_trees": best_trees}),
                "seconds": round(dur, 2),
                "pr_auc": trial_res["pr_auc"],
                "recall_top10": trial_res["recall_top10"],
                "precision_top5": trial_res["precision_top5"],
                "brier": trial_res["brier"],
            },
            runs_path=runs_csv_path,
        )

        trial_records.append(
            {
                "trial": idx,
                **cfg,
                "best_trees": best_trees,
                "pr_auc": trial_res["pr_auc"],
                "recall_top10": trial_res["recall_top10"],
                "precision_top5": trial_res["precision_top5"],
                "brier": trial_res["brier"],
                "seconds": round(dur, 2),
            }
        )

        if float(trial_res["pr_auc"]) > best_pr:
            best_pr = float(trial_res["pr_auc"])
            best_model = model
            best_cfg = {**cfg, "best_trees": best_trees}
            best_scores = scores
            best_trees_win = best_trees

    assert best_model is not None and best_cfg is not None and best_scores is not None
    eval_df["score"] = best_scores
    best_res = evaluate(eval_df, "new_fire_next_48h", "score", model="xgb_v1_tuned", split="val", set_name="primary")
    best_res["set_name"] = set_name
    record_result(best_res, csv_path=results_csv_path, md_path=results_md_path)

    if model_save_path is not None:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        best_model.save_model(model_save_path)

    trials_df = pd.DataFrame(trial_records).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    trials_df.attrs["best_trees"] = best_trees_win
    return best_model, best_cfg, trials_df, best_scores


def run_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_keys: pd.DataFrame,
    feature_group_name: str = "G3",
    n_estimators: int = 200,
    min_samples_leaf: int = 50,
    max_samples: float | None = 0.1,
    partial: bool = False,
    n_cells: int | None = None,
    seed: int = 42,
    runs_csv_path: Path = RUNS_CSV_PATH,
    results_csv_path: Path = RESULTS_CSV_PATH,
    results_md_path: Path = RESULTS_MD_PATH,
) -> tuple[RandomForestClassifier, dict[str, Any], np.ndarray]:
    """Fit Random Forest classifier benchmark on train split and score 2023 val."""
    assert_test_split_locked()
    set_name = _format_set_name("primary", partial=partial, n_cells=n_cells)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    scores = rf.predict_proba(X_val)[:, 1]
    dur = time.perf_counter() - t0

    eval_df = val_keys.copy()
    eval_df["new_fire_next_48h"] = y_val.values
    eval_df["score"] = scores
    res = evaluate(eval_df, "new_fire_next_48h", "score", model="rf_v1", split="val", set_name="primary")
    res["set_name"] = set_name
    record_result(res, csv_path=results_csv_path, md_path=results_md_path)

    log_run(
        {
            "run_id": "v1_rf_01",
            "date": str(date.today()),
            "model": "rf_v1",
            "feature_group": feature_group_name,
            "label": "new_fire_next_48h",
            "set_name": set_name,
            "parameters": json.dumps(
                {
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "max_samples": max_samples,
                    "max_features": "sqrt",
                    "class_weight": "balanced_subsample",
                }
            ),
            "seconds": round(dur, 2),
            "pr_auc": res["pr_auc"],
            "recall_top10": res["recall_top10"],
            "precision_top5": res["precision_top5"],
            "brier": res["brier"],
        },
        runs_path=runs_csv_path,
    )
    return rf, res, scores


def compute_feature_importance(
    model: xgb.XGBClassifier | RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_sample: int = 200_000,
    n_repeats: int = 3,
    seed: int = 42,
    output_path: Path = FEATURE_IMPORTANCE_CSV_PATH,
) -> pd.DataFrame:
    """Compute native gain importance and validation permutation PR-AUC importance."""
    rng = np.random.default_rng(seed)
    n_rows = len(X_val)
    if n_rows > n_sample:
        idx = np.sort(rng.choice(n_rows, size=n_sample, replace=False))
        X_sub = X_val.iloc[idx].reset_index(drop=True)
        y_sub = y_val.iloc[idx].reset_index(drop=True)
    else:
        X_sub = X_val
        y_sub = y_val

    gain_map: dict[str, float] = {f: 0.0 for f in feature_names}
    if isinstance(model, xgb.XGBClassifier):
        raw_gain = model.get_booster().get_score(importance_type="gain")
        for f in feature_names:
            gain_map[f] = float(raw_gain.get(f, 0.0))
    elif hasattr(model, "feature_importances_"):
        for f, imp in zip(feature_names, model.feature_importances_):
            gain_map[f] = float(imp)

    perm = permutation_importance(
        model,
        X_sub[feature_names],
        y_sub,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": [gain_map[f] for f in feature_names],
            "perm_importance_mean": perm.importances_mean.astype(float),
            "perm_importance_std": perm.importances_std.astype(float),
        }
    ).sort_values("perm_importance_mean", ascending=False, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(output_path, index=False)
    return df_imp


def save_xgb_model(model: xgb.XGBClassifier, model_path: Path) -> Path:
    """Save XGBClassifier natively to JSON format."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    return model_path


def load_xgb_model(model_path: Path) -> xgb.XGBClassifier:
    """Load XGBClassifier from native JSON format."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model
