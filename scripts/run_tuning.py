"""CLI entry point for Model v1 ablation, tuning, Random Forest benchmark, and feature importance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402

from src.features.build_features import ALL_FEATURE_NAMES  # noqa: E402
from src.models.features_io import FEATURES_FULL_PATH, load_features  # noqa: E402
from src.models.train_v1 import (  # noqa: E402
    DEFAULT_XGB_PARAMS,
    FEATURE_GROUPS,
    FEATURES_PARTIAL_PATH,
    assert_test_split_locked,
    compute_feature_importance,
    fit_xgb_two_stage,
    load_partial_cells,
    run_ablation,
    run_random_forest,
    run_xgb_tuning,
    save_xgb_model,
    select_winner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model v1 Day 5 pipeline (ablation, tuning, RF, importance).")
    parser.add_argument("--partial", action="store_true", help="Run on features_partial.parquet using partial_cells.json.")
    parser.add_argument("--ablation-only", action="store_true", help="Stop after running G1/G2/G3 ablation.")
    parser.add_argument("--primary-only", action="store_true", help="Run ablation on primary label new_fire_next_48h only.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random search trials for XGBoost.")
    parser.add_argument("--max-minutes", type=float, default=60.0, help="Wall-clock cap in minutes for tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    assert_test_split_locked()
    args = parse_args()
    f_path: Path = FEATURES_PARTIAL_PATH if args.partial else FEATURES_FULL_PATH
    partial_cells = load_partial_cells() if args.partial else None
    n_cells = len(partial_cells) if partial_cells is not None else None

    print("=== Step 1: Feature Group Ablation (G1, G2, G3) ===")
    abl_df = run_ablation(partial=args.partial, features_path=f_path, primary_only=args.primary_only, seed=args.seed)
    print(abl_df.to_string(index=False))
    if args.ablation_only:
        return

    # Choose best feature group on primary label new_fire_next_48h (prefer simpler if tied within 0.002)
    prim_abl = abl_df[(abl_df["label"] == "new_fire_next_48h") & (abl_df["group"].isin(["G1", "G2", "G3"]))]
    best_group_row = prim_abl.sort_values(["pr_auc", "n_features"], ascending=[False, True]).iloc[0]
    best_group = str(best_group_row["group"])
    feat_cols = FEATURE_GROUPS[best_group]
    print(f"\nSelected feature group for tuning: {best_group} ({len(feat_cols)} features)")

    X_tr, y_tr, keys_tr = load_features(
        "train",
        label="new_fire_next_48h",
        feature_names=feat_cols,
        primary=True,
        features_path=f_path,
        allowed_cells=partial_cells,
    )
    X_va, y_va, keys_va = load_features(
        "val",
        label="new_fire_next_48h",
        feature_names=feat_cols,
        primary=True,
        features_path=f_path,
        allowed_cells=partial_cells,
    )

    print(f"\n=== Step 2: Random Search Tuning ({args.trials} trials, cap {args.max_minutes} min) ===")
    xgb_model, best_cfg, trials_df, _ = run_xgb_tuning(
        X_train=X_tr,
        y_train=y_tr,
        train_dates=keys_tr["date"],
        X_val=X_va,
        y_val=y_va,
        val_keys=keys_va,
        feature_group_name=best_group,
        n_trials=args.trials,
        max_minutes=args.max_minutes,
        partial=args.partial,
        n_cells=n_cells,
        seed=args.seed,
        model_save_path=PROJECT_ROOT / "models" / "xgb_v1_tuned" / "model.json",
    )
    print(trials_df.head(5).to_string(index=False))

    print("\n=== Step 3: Random Forest Benchmark ===")
    rf_model, rf_res, _ = run_random_forest(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        val_keys=keys_va,
        feature_group_name=best_group,
        partial=args.partial,
        n_cells=n_cells,
        seed=args.seed,
    )
    print(f"rf_v1: PR-AUC={rf_res['pr_auc']:.4f}, Recall@10%={rf_res['recall_top10']:.4f}")

    best_trial_row = trials_df.iloc[0]
    candidates = [
        {
            "model": f"xgb_v1_{best_group}_default",
            "pr_auc": float(best_group_row["pr_auc"]),
            "recall_top10": float(best_group_row["recall_top10"]),
            "complexity": 1,
        },
        {
            "model": "xgb_v1_tuned",
            "pr_auc": float(best_trial_row["pr_auc"]),
            "recall_top10": float(best_trial_row["recall_top10"]),
            "complexity": 2,
        },
        {
            "model": "rf_v1",
            "pr_auc": float(rf_res["pr_auc"]),
            "recall_top10": float(rf_res["recall_top10"]),
            "complexity": 3,
        },
    ]
    winner = select_winner(candidates)
    print(f"\n=== Winner: {winner['model']} (PR-AUC={winner['pr_auc']:.4f}, Recall@10%={winner['recall_top10']:.4f}) ===")

    if winner["model"] == "xgb_v1_tuned":
        save_xgb_model(xgb_model, PROJECT_ROOT / "models" / "xgb_v1_winner" / "model.json")
        imp_source = xgb_model
    elif winner["model"] == "rf_v1":
        rf_path = PROJECT_ROOT / "models" / "rf_v1_winner" / "model.joblib"
        rf_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_model, rf_path)
        imp_source = rf_model
    else:
        default_xgb, _, _ = fit_xgb_two_stage(
            X_train=X_tr,
            y_train=y_tr,
            train_dates=keys_tr["date"],
            X_val=X_va,
            params=DEFAULT_XGB_PARAMS,
            max_trees=600,
            seed=args.seed,
        )
        save_xgb_model(default_xgb, PROJECT_ROOT / "models" / "xgb_v1_winner" / "model.json")
        imp_source = default_xgb

    print("\n=== Step 4: Feature Importance ===")
    imp_df = compute_feature_importance(imp_source, X_va, y_va, feature_names=feat_cols, seed=args.seed)
    print(imp_df.head(10).to_string(index=False))
    print(f"\nAll feature count in registry/G3: {len(ALL_FEATURE_NAMES)}")
    print(f"Winner best params: {best_cfg}")


if __name__ == "__main__":
    main()
