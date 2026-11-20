"""Phase C: Shared XGBoost hyperparameter tuning (20 trials on G3).

Runs deterministic random search with 75-minute wall-clock cap.
Uses two-stage training protocol on full 1,167-cell features.parquet.
Supports resuming from already logged trials in docs/runs.csv.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from src.config import DOCS_AUDIT_DIR, PROCESSED_DIR  # noqa: E402
from src.features.build_features import FEATURES_FULL_PATH  # noqa: E402
from src.models.evaluate import evaluate, record_result  # noqa: E402
from src.models.features_io import load_features  # noqa: E402
from src.models.train_v0 import log_run  # noqa: E402
from src.models.train_v1 import (  # noqa: E402
    DEFAULT_XGB_PARAMS,
    FEATURE_GROUPS,
    RESULTS_CSV_PATH,
    RESULTS_MD_PATH,
    RUNS_CSV_PATH,
    assert_test_split_locked,
    fit_xgb_two_stage,
    sample_tuning_configs,
)


def load_existing_trials(runs_csv_path: Path) -> dict[int, dict[str, Any]]:
    """Parse existing tuning runs from docs/runs.csv to support resumption."""
    if not runs_csv_path.exists():
        return {}
    existing: dict[int, dict[str, Any]] = {}
    df = pd.read_csv(runs_csv_path)
    for _, row in df.iterrows():
        run_id = str(row["run_id"])
        match = re.match(r"^v1_tune_(\d+)$", run_id)
        if match and str(row.get("label")) == "new_fire_next_48h" and str(row.get("set_name")) == "primary":
            t_idx = int(match.group(1))
            params_raw = row["parameters"]
            params = json.loads(params_raw) if isinstance(params_raw, str) else {}
            best_trees = params.pop("best_trees", 0)
            existing[t_idx] = {
                "trial": t_idx,
                **params,
                "best_trees": best_trees,
                "pr_auc": float(row["pr_auc"]),
                "recall_top10": float(row["recall_top10"]),
                "precision_top5": float(row["precision_top5"]),
                "brier": float(row["brier"]),
                "seconds": float(row["seconds"]),
            }
    return existing


def phase_c() -> dict:
    """Run Phase C: hyperparameter tuning."""
    t0 = time.perf_counter()
    assert_test_split_locked()

    # Load cells_final
    cells_final_path = PROCESSED_DIR / "cells_final.json"
    with open(cells_final_path, encoding="utf-8") as f:
        cells_final = json.load(f)
    n_cells = cells_final["count"]
    print(f"Full-cell run: {n_cells} cells")

    # Load G3 features for train and val splits
    g3_features = FEATURE_GROUPS["G3"]
    print(f"G3 features ({len(g3_features)}): {g3_features}")

    print("\nLoading train split...")
    X_train, y_train, keys_train = load_features(
        "train",
        label="new_fire_next_48h",
        feature_names=g3_features,
        primary=True,
        features_path=FEATURES_FULL_PATH,
    )
    print(f"  Train: {len(X_train):,} rows, {int(y_train.sum()):,} positives ({100*y_train.mean():.2f}%)")

    print("Loading val split...")
    X_val, y_val, keys_val = load_features(
        "val",
        label="new_fire_next_48h",
        feature_names=g3_features,
        primary=True,
        features_path=FEATURES_FULL_PATH,
    )
    print(f"  Val:   {len(X_val):,} rows, {int(y_val.sum()):,} positives ({100*y_val.mean():.2f}%)")

    configs = sample_tuning_configs(n_trials=20, seed=42)
    existing_trials = load_existing_trials(RUNS_CSV_PATH)
    if existing_trials:
        print(f"\nFound {len(existing_trials)} previously completed trials in {RUNS_CSV_PATH}:")
        for tid in sorted(existing_trials):
            print(f"  Trial {tid:02d}: PR-AUC={existing_trials[tid]['pr_auc']:.6f}, "
                  f"trees={existing_trials[tid]['best_trees']}, {existing_trials[tid]['seconds']:.1f}s")

    eval_df = keys_val.copy()
    eval_df["new_fire_next_48h"] = y_val.values

    trial_records: list[dict[str, Any]] = []
    best_model: xgb.XGBClassifier | None = None
    best_cfg: dict[str, Any] | None = None
    best_scores: Any = None
    best_pr = -1.0
    best_trial_idx = 0

    accumulated_prev_seconds = sum(t["seconds"] for t in existing_trials.values())
    max_minutes = 75.0

    print("\n--- RUNNING TUNING TRIALS (Target 20 trials, 75-min cap) ---")
    for idx, cfg in enumerate(configs, start=1):
        if idx in existing_trials:
            rec = existing_trials[idx]
            trial_records.append(rec)
            if rec["pr_auc"] > best_pr:
                best_pr = rec["pr_auc"]
                best_cfg = {**cfg, "best_trees": rec["best_trees"]}
                best_trial_idx = idx
            continue

        elapsed_total_min = (accumulated_prev_seconds + (time.perf_counter() - t0)) / 60.0
        if idx > 1 and elapsed_total_min >= max_minutes:
            print(f"Time limit reached ({elapsed_total_min:.1f} min >= {max_minutes} min). Stopping trials.")
            break

        print(f"\nStarting Trial {idx:02d}/20: {cfg}")
        t_trial = time.perf_counter()
        model, best_trees, scores = fit_xgb_two_stage(
            X_train=X_train,
            y_train=y_train,
            train_dates=keys_train["date"],
            X_val=X_val,
            params=cfg,
            max_trees=1000,
            seed=42,
        )
        dur = time.perf_counter() - t_trial
        eval_df["score"] = scores
        trial_res = evaluate(eval_df, "new_fire_next_48h", "score",
                             model=f"xgb_tune_t{idx:02d}", split="val", set_name="primary")

        log_run(
            {
                "run_id": f"v1_tune_{idx:02d}",
                "date": str(date.today()),
                "model": f"xgb_v1_tune_{idx:02d}",
                "feature_group": "G3",
                "label": "new_fire_next_48h",
                "set_name": "primary",
                "parameters": json.dumps({**cfg, "best_trees": best_trees}),
                "seconds": round(dur, 2),
                "pr_auc": trial_res["pr_auc"],
                "recall_top10": trial_res["recall_top10"],
                "precision_top5": trial_res["precision_top5"],
                "brier": trial_res["brier"],
            },
            runs_path=RUNS_CSV_PATH,
        )

        rec = {
            "trial": idx,
            **cfg,
            "best_trees": best_trees,
            "pr_auc": trial_res["pr_auc"],
            "recall_top10": trial_res["recall_top10"],
            "precision_top5": trial_res["precision_top5"],
            "brier": trial_res["brier"],
            "seconds": round(dur, 2),
        }
        trial_records.append(rec)
        print(f"  Trial {idx:02d} complete in {dur:.1f}s: PR-AUC={trial_res['pr_auc']:.6f}, "
              f"R@10%={trial_res['recall_top10']:.4f}, trees={best_trees}")

        if float(trial_res["pr_auc"]) > best_pr:
            best_pr = float(trial_res["pr_auc"])
            best_model = model
            best_cfg = {**cfg, "best_trees": best_trees}
            best_scores = scores
            best_trial_idx = idx

    assert best_cfg is not None

    # If the winning trial was an existing trial (e.g. from a prior run), fit it to get its scores & model
    if best_model is None or best_scores is None:
        print(f"\nWinning trial was Trial {best_trial_idx:02d} (from previous run). Fitting winning model for evaluation...")
        best_model, _, best_scores = fit_xgb_two_stage(
            X_train=X_train,
            y_train=y_train,
            train_dates=keys_train["date"],
            X_val=X_val,
            params={k: v for k, v in best_cfg.items() if k != "best_trees"},
            max_trees=best_cfg["best_trees"],
            early_stopping_rounds=9999,  # keep exact trees
            seed=42,
        )

    eval_df["score"] = best_scores
    best_res = evaluate(eval_df, "new_fire_next_48h", "score", model="xgb_v1_tuned", split="val", set_name="primary")
    best_res["set_name"] = "primary"
    record_result(best_res, csv_path=RESULTS_CSV_PATH, md_path=RESULTS_MD_PATH)

    trials_df = pd.DataFrame(trial_records).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    # Save trials CSV
    trials_csv = DOCS_AUDIT_DIR / "tuning_trials.csv"
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(trials_csv, index=False)
    print(f"\nSaved {len(trials_df)} trials to {trials_csv}")

    # Compare vs default G3
    print("\nRunning default G3 for comparison...")
    _, default_trees, default_scores = fit_xgb_two_stage(
        X_train, y_train, keys_train["date"], X_val,
        params=DEFAULT_XGB_PARAMS, max_trees=600, seed=42,
    )
    eval_df["score"] = default_scores
    default_res = evaluate(eval_df, "new_fire_next_48h", "score",
                           model="xgb_v1_G3_default", split="val", set_name="primary")

    default_prauc = float(default_res["pr_auc"])
    tuned_prauc = float(best_res["pr_auc"])
    margin = tuned_prauc - default_prauc
    print(f"Default G3 PR-AUC: {default_prauc:.6f} (trees={default_trees})")
    print(f"Tuned G3 PR-AUC:   {tuned_prauc:.6f} (Trial {best_trial_idx:02d}, trees={best_cfg['best_trees']})")
    print(f"Margin:            {margin:+.6f}")

    print("\n--- ALL TRIALS RANKED BY PR-AUC ---")
    print(trials_df[["trial", "max_depth", "learning_rate", "min_child_weight", "subsample",
                     "colsample_bytree", "reg_lambda", "best_trees", "pr_auc", "recall_top10", "seconds"]].to_string(index=False))

    elapsed = time.perf_counter() - t0
    result = {
        "status": "PASS",
        "trials_completed": len(trials_df),
        "best_trial": best_trial_idx,
        "best_pr_auc": round(tuned_prauc, 6),
        "default_pr_auc": round(default_prauc, 6),
        "margin": round(margin, 6),
        "best_config": best_cfg,
        "default_trees": default_trees,
        "elapsed_s": round(elapsed, 1),
    }
    return result


if __name__ == "__main__":
    result = phase_c()
    print(f"\n=== Phase C result: {json.dumps(result, indent=2)} ===")
