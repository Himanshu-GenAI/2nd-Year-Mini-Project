"""Phase D: Full-cell ablation, bootstrap intervals, rolling origin, RF, winner selection.

Steps:
1. Full ablation: climatology, G1, G1K, G2, G2R, G3 (default) and G3_tuned (two-stage)
   + sensitivity check with fixed 150 trees.
2. Block bootstrap: 1,000 resamples across 6 contrasts.
3. Rolling origin: 2021 and 2022 for climatology, G1, G2, G3.
4. Random Forest benchmark: G3, 200 trees.
5. Secondary label: winner on fire_next_48h (all rows).
6. Winner selection per D18 rule.
7. Feature importance: gain + permutation (200k sample, 3 repeats).
8. Write docs/audit/full_run_diagnostics.md.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.config import DOCS_AUDIT_DIR, FIRE_TABLE_PATH  # noqa: E402
from src.features.build_features import ALL_FEATURE_NAMES, FEATURES_FULL_PATH  # noqa: E402
from src.models.baselines import fit_climatology, predict_climatology  # noqa: E402
from src.models.diagnostics import (  # noqa: E402
    DIAGNOSTIC_FEATURE_GROUPS,
    get_calendar_weeks,
)
from src.models.evaluate import (  # noqa: E402
    evaluate,
    recall_at_top_frac,
    record_result,
)
from src.models.features_io import load_features  # noqa: E402
from src.models.train_v1 import (  # noqa: E402
    DEFAULT_XGB_PARAMS,
    FEATURE_GROUPS,
    assert_test_split_locked,
    compute_feature_importance,
    fit_xgb_two_stage,
    run_random_forest,
    select_winner,
)


def load_tuned_config() -> dict[str, Any]:
    """Load tuned hyperparameters from Phase C output."""
    trials_csv = DOCS_AUDIT_DIR / "tuning_trials.csv"
    if not trials_csv.exists():
        raise FileNotFoundError(f"Phase C output not found: {trials_csv}")
    trials_df = pd.read_csv(trials_csv)
    best_row = trials_df.iloc[0]
    cfg = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "min_child_weight": float(best_row["min_child_weight"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "reg_lambda": float(best_row["reg_lambda"]),
        "best_trees": int(best_row["best_trees"]),
    }
    return cfg


def step1_ablation(tuned_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run full ablation (two-stage) and sensitivity check (fixed 150 trees)."""
    print("\n" + "=" * 70)
    print("STEP 1: Full Ablation & Sensitivity Check (1,167 cells)")
    print("=" * 70)
    assert_test_split_locked()

    label = "new_fire_next_48h"
    ablation_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []

    print("Loading full features for train and val splits (primary set)...")
    X_tr, y_tr, keys_tr = load_features(
        "train", label=label, feature_names=ALL_FEATURE_NAMES,
        primary=True, features_path=FEATURES_FULL_PATH,
    )
    X_va, y_va, keys_va = load_features(
        "val", label=label, feature_names=ALL_FEATURE_NAMES,
        primary=True, features_path=FEATURES_FULL_PATH,
    )
    print(f"  Train: {len(X_tr):,} rows ({int(y_tr.sum()):,} pos)")
    print(f"  Val:   {len(X_va):,} rows ({int(y_va.sum()):,} pos)")

    preds_df = keys_va[["cell_id", "date"]].copy()
    preds_df[label] = y_va.values

    # 1. Climatology baseline
    fire_df = pd.read_parquet(FIRE_TABLE_PATH)
    train_fire = fire_df[fire_df["split"] == "train"].copy()
    val_fire = fire_df[fire_df["split"] == "val"].copy()
    val_primary = val_fire[~val_fire["burning_prev3"].astype(bool)].copy()

    clim = fit_climatology(train_fire, label=label, alpha=1.0)
    clim_scores = predict_climatology(clim, val_primary)
    preds_df["score_climatology"] = clim_scores

    eval_clim = keys_va.copy()
    eval_clim[label] = y_va.values
    eval_clim["score"] = clim_scores
    res_clim = evaluate(eval_clim, label, "score", model="climatology", split="val", set_name="primary")
    res_clim["recall_top10"] = round(float(recall_at_top_frac(eval_clim, label, "score", frac=0.10)[0]), 4)
    record_result(res_clim)
    print(f"  climatology: PR-AUC={res_clim['pr_auc']:.6f}, R@10%={res_clim['recall_top10']:.4f}")

    ablation_rows.append({
        "group": "climatology", "params": "N/A", "n_features": 0,
        "pr_auc": res_clim["pr_auc"], "recall_top10": res_clim["recall_top10"],
        "precision_top5": res_clim["precision_top5"], "brier": res_clim["brier"],
        "best_trees": 0, "seconds": 0.0,
    })
    fixed_rows.append({
        "group": "climatology", "trees": 0,
        "pr_auc": res_clim["pr_auc"], "recall_top10": res_clim["recall_top10"],
        "precision_top5": res_clim["precision_top5"], "brier": res_clim["brier"],
    })

    # 2. Ablation groups: G1, G1K, G2, G2R, G3 (default)
    groups = ["G1", "G1K", "G2", "G2R", "G3"]
    models: dict[str, Any] = {}

    for g_name in groups:
        cols = DIAGNOSTIC_FEATURE_GROUPS[g_name]
        print(f"\n--- Training {g_name} (default, {len(cols)} features) ---")
        t0 = time.perf_counter()
        m_ts, best_trees, scores_ts = fit_xgb_two_stage(
            X_tr[cols], y_tr, keys_tr["date"], X_va[cols],
            params=DEFAULT_XGB_PARAMS, max_trees=600, early_stopping_rounds=30, seed=42,
        )
        dur_ts = time.perf_counter() - t0
        preds_df[f"score_{g_name}"] = scores_ts

        eval_ts = keys_va.copy()
        eval_ts[label] = y_va.values
        eval_ts["score"] = scores_ts
        res_ts = evaluate(eval_ts, label, "score", model=f"xgb_v1_{g_name}_default", split="val", set_name="primary")
        res_ts["recall_top10"] = round(float(recall_at_top_frac(eval_ts, label, "score", frac=0.10)[0]), 4)
        record_result(res_ts)
        print(f"  {g_name} default: PR-AUC={res_ts['pr_auc']:.6f}, R@10%={res_ts['recall_top10']:.4f}, trees={best_trees} in {dur_ts:.1f}s")

        ablation_rows.append({
            "group": g_name, "params": "default", "n_features": len(cols),
            "pr_auc": res_ts["pr_auc"], "recall_top10": res_ts["recall_top10"],
            "precision_top5": res_ts["precision_top5"], "brier": res_ts["brier"],
            "best_trees": best_trees, "seconds": round(dur_ts, 2),
        })

        if g_name == "G3":
            models["G3_default"] = (m_ts, best_trees, scores_ts, res_ts)

        # Sensitivity Check: Fixed 150 trees refit (Stage 2 fit only, no early stopping)
        t_fix = time.perf_counter()
        m_fixed = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=int(DEFAULT_XGB_PARAMS["max_depth"]),
            learning_rate=float(DEFAULT_XGB_PARAMS["learning_rate"]),
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
        )
        m_fixed.fit(X_tr[cols], y_tr, verbose=False)
        scores_fixed = m_fixed.predict_proba(X_va[cols])[:, 1]
        dur_fix = time.perf_counter() - t_fix

        eval_fixed = keys_va.copy()
        eval_fixed[label] = y_va.values
        eval_fixed["score"] = scores_fixed
        res_fixed = evaluate(eval_fixed, label, "score", model=f"xgb_v1_{g_name}_fixed150", split="val", set_name="primary")
        res_fixed["recall_top10"] = round(float(recall_at_top_frac(eval_fixed, label, "score", frac=0.10)[0]), 4)
        print(f"  {g_name} fixed 150: PR-AUC={res_fixed['pr_auc']:.6f}, R@10%={res_fixed['recall_top10']:.4f} in {dur_fix:.1f}s")

        fixed_rows.append({
            "group": g_name, "trees": 150,
            "pr_auc": res_fixed["pr_auc"], "recall_top10": res_fixed["recall_top10"],
            "precision_top5": res_fixed["precision_top5"], "brier": res_fixed["brier"],
        })

    # 3. G3 Tuned
    print(f"\n--- Training G3_tuned ({len(ALL_FEATURE_NAMES)} features) ---")
    g3_tuned_params = {k: v for k, v in tuned_cfg.items() if k != "best_trees"}
    t0_t = time.perf_counter()
    m_tuned, best_trees_t, scores_tuned = fit_xgb_two_stage(
        X_tr[ALL_FEATURE_NAMES], y_tr, keys_tr["date"], X_va[ALL_FEATURE_NAMES],
        params=g3_tuned_params, max_trees=1000, early_stopping_rounds=30, seed=42,
    )
    dur_t = time.perf_counter() - t0_t
    preds_df["score_G3_tuned"] = scores_tuned

    eval_tuned = keys_va.copy()
    eval_tuned[label] = y_va.values
    eval_tuned["score"] = scores_tuned
    res_tuned = evaluate(eval_tuned, label, "score", model="xgb_v1_G3_tuned", split="val", set_name="primary")
    res_tuned["recall_top10"] = round(float(recall_at_top_frac(eval_tuned, label, "score", frac=0.10)[0]), 4)
    record_result(res_tuned)
    print(f"  G3 tuned: PR-AUC={res_tuned['pr_auc']:.6f}, R@10%={res_tuned['recall_top10']:.4f}, trees={best_trees_t} in {dur_t:.1f}s")

    ablation_rows.append({
        "group": "G3", "params": "tuned", "n_features": len(ALL_FEATURE_NAMES),
        "pr_auc": res_tuned["pr_auc"], "recall_top10": res_tuned["recall_top10"],
        "precision_top5": res_tuned["precision_top5"], "brier": res_tuned["brier"],
        "best_trees": best_trees_t, "seconds": round(dur_t, 2),
    })
    models["G3_tuned"] = (m_tuned, best_trees_t, scores_tuned, res_tuned)

    ablation_df = pd.DataFrame(ablation_rows)
    fixed_df = pd.DataFrame(fixed_rows)

    DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ablation_csv = DOCS_AUDIT_DIR / "full_run_ablation.csv"
    fixed_csv = DOCS_AUDIT_DIR / "full_run_fixed150_sensitivity.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    fixed_df.to_csv(fixed_csv, index=False)
    print(f"\nSaved ablation to {ablation_csv} and fixed-150 to {fixed_csv}")

    context = {
        "preds_df": preds_df,
        "models": models,
        "res_clim": res_clim,
        "X_va": X_va,
        "y_va": y_va,
        "keys_va": keys_va,
    }
    return ablation_df, fixed_df, context


def step2_bootstrap(preds: pd.DataFrame, label: str = "new_fire_next_48h") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Block bootstrap by calendar week of 2023 for 6 contrasts (1,000 resamples)."""
    print("\n" + "=" * 70)
    print("STEP 2: Block Bootstrap by Calendar Week (1,000 resamples, seed 42)")
    print("=" * 70)
    assert_test_split_locked()

    y_true = preds[label].values.astype(int)
    score_cols = {
        "climatology": preds["score_climatology"].values,
        "G1": preds["score_G1"].values,
        "G1K": preds["score_G1K"].values,
        "G2": preds["score_G2"].values,
        "G2R": preds["score_G2R"].values,
        "G3": preds["score_G3"].values,
    }

    weeks = get_calendar_weeks(preds["date"]).values
    unique_weeks = np.unique(weeks)
    n_unique_weeks = len(unique_weeks)
    week_indices = {w: np.where(weeks == w)[0] for w in unique_weeks}

    rng = np.random.default_rng(42)

    contrasts = [
        ("G2 - G1", "G2", "G1"),
        ("G3 - G2", "G3", "G2"),
        ("G1K - G1", "G1K", "G1"),
        ("G2R - G2", "G2R", "G2"),
        ("G3 - climatology", "G3", "climatology"),
        ("G3 - G1", "G3", "G1"),
    ]

    replicates = {name: [] for name, _, _ in contrasts}
    raw_draws = []

    for b in range(1000):
        sampled_weeks = rng.choice(unique_weeks, size=n_unique_weeks, replace=True)
        idx_b = np.concatenate([week_indices[w] for w in sampled_weeks])
        y_b = y_true[idx_b]
        if y_b.sum() == 0:
            continue

        pr_scores = {}
        for key, scores in score_cols.items():
            pr_scores[key] = float(average_precision_score(y_b, scores[idx_b]))

        draw_row: dict[str, Any] = {"replicate": b + 1}
        for name, a, b_key in contrasts:
            diff_val = pr_scores[a] - pr_scores[b_key]
            replicates[name].append(diff_val)
            draw_row[name] = round(diff_val, 6)
        raw_draws.append(draw_row)

    point_pr = {}
    for key, scores in score_cols.items():
        point_pr[key] = float(average_precision_score(y_true, scores))

    summary_rows = []
    for name, a, b_key in contrasts:
        vals = np.array(replicates[name])
        ci_low = float(np.percentile(vals, 2.5))
        ci_high = float(np.percentile(vals, 97.5))
        point_diff = point_pr[a] - point_pr[b_key]
        inc_zero = bool(ci_low <= 0.0 <= ci_high)
        summary_rows.append({
            "contrast": name,
            "point_diff": round(point_diff, 6),
            "boot_mean": round(float(np.mean(vals)), 6),
            "boot_std": round(float(np.std(vals, ddof=1)), 6),
            "ci_lower_95": round(ci_low, 6),
            "ci_upper_95": round(ci_high, 6),
            "includes_zero": inc_zero,
            "verdict": "inconclusive" if inc_zero else "significant",
        })
        print(f"  {name:25s}: {point_diff:+.6f}  95% CI [{ci_low:+.6f}, {ci_high:+.6f}] "
              f"{'includes 0 -> inconclusive' if inc_zero else 'excludes 0 -> significant'}")

    summary_df = pd.DataFrame(summary_rows)
    draws_df = pd.DataFrame(raw_draws)

    boot_csv = DOCS_AUDIT_DIR / "full_run_bootstrap_summary.csv"
    draws_csv = DOCS_AUDIT_DIR / "full_run_bootstrap.csv"
    summary_df.to_csv(boot_csv, index=False)
    draws_df.to_csv(draws_csv, index=False)
    print(f"\nSaved bootstrap summary to {boot_csv} and draws to {draws_csv}")
    return summary_df, draws_df


def step3_rolling_origin() -> pd.DataFrame:
    """Rolling-origin evaluation for validation years 2021 and 2022."""
    print("\n" + "=" * 70)
    print("STEP 3: Rolling-Origin Repeat (2021 and 2022)")
    print("=" * 70)
    assert_test_split_locked()

    all_rows = []
    label = "new_fire_next_48h"

    X_all, y_all, keys_all = load_features(
        "train", label=label, feature_names=ALL_FEATURE_NAMES,
        primary=True, features_path=FEATURES_FULL_PATH,
    )
    dates = pd.to_datetime(keys_all["date"])

    for val_year in [2021, 2022]:
        s1_es_year = val_year - 1
        s1_train_end_year = s1_es_year - 1

        train_mask = dates.dt.year <= s1_es_year
        val_mask = dates.dt.year == val_year

        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]
        keys_tr = keys_all[train_mask]
        X_va = X_all[val_mask]
        y_va = y_all[val_mask]
        keys_va = keys_all[val_mask]

        print(f"\n--- Val Year {val_year} (Train <= {s1_es_year}: {len(X_tr):,} rows | Val = {val_year}: {len(X_va):,} rows, {int(y_va.sum()):,} pos) ---")

        # Climatology
        clim_train_df = keys_tr[["cell_id", "date"]].copy()
        clim_train_df[label] = y_tr.values
        clim_train_df["burning_prev3"] = False
        clim = fit_climatology(clim_train_df, label=label, alpha=1.0)

        eval_clim = keys_va[["cell_id", "date", "burning_prev3", "split"]].copy()
        eval_clim[label] = y_va.values
        eval_clim["score"] = predict_climatology(clim, eval_clim)
        res_clim = evaluate(eval_clim, label, "score", model="climatology", split=f"val_{val_year}", set_name="primary")
        res_clim["recall_top10"] = round(float(recall_at_top_frac(eval_clim, label, "score", frac=0.10)[0]), 4)

        all_rows.append({
            "val_year": val_year, "group": "climatology", "n_features": 0,
            "pr_auc": res_clim["pr_auc"], "recall_top10": res_clim["recall_top10"],
            "precision_top5": res_clim["precision_top5"], "brier": res_clim["brier"],
            "best_trees": 0,
        })
        print(f"  climatology: PR-AUC={res_clim['pr_auc']:.6f}, R@10%={res_clim['recall_top10']:.4f}")

        for g_name in ["G1", "G2", "G3"]:
            cols = FEATURE_GROUPS[g_name]
            _, best_trees, scores = fit_xgb_two_stage(
                X_tr[cols], y_tr, keys_tr["date"], X_va[cols],
                params=DEFAULT_XGB_PARAMS, max_trees=600, early_stopping_rounds=30, seed=42,
                s1_train_end_year=s1_train_end_year, s1_es_year=s1_es_year,
            )

            eval_df = keys_va.copy()
            eval_df[label] = y_va.values
            eval_df["score"] = scores
            res = evaluate(eval_df, label, "score", model=f"xgb_v1_{g_name}", split=f"val_{val_year}", set_name="primary")
            res["recall_top10"] = round(float(recall_at_top_frac(eval_df, label, "score", frac=0.10)[0]), 4)

            all_rows.append({
                "val_year": val_year, "group": g_name, "n_features": len(cols),
                "pr_auc": res["pr_auc"], "recall_top10": res["recall_top10"],
                "precision_top5": res["precision_top5"], "brier": res["brier"],
                "best_trees": best_trees,
            })
            print(f"  {g_name:11s}: PR-AUC={res['pr_auc']:.6f}, R@10%={res['recall_top10']:.4f}, trees={best_trees}")

    rolling_df = pd.DataFrame(all_rows)
    rolling_csv = DOCS_AUDIT_DIR / "full_run_rolling.csv"
    rolling_df.to_csv(rolling_csv, index=False)
    print(f"\nSaved rolling origin to {rolling_csv}")
    return rolling_df


def step4_random_forest(context: dict[str, Any]) -> dict[str, Any]:
    """Run Random Forest benchmark on G3 (200 trees, 25-min cap)."""
    print("\n" + "=" * 70)
    print("STEP 4: Random Forest Benchmark (G3, 200 trees)")
    print("=" * 70)
    assert_test_split_locked()

    label = "new_fire_next_48h"
    g3_cols = FEATURE_GROUPS["G3"]

    X_tr, y_tr, _ = load_features(
        "train", label=label, feature_names=g3_cols,
        primary=True, features_path=FEATURES_FULL_PATH,
    )
    X_va = context["X_va"][g3_cols]
    y_va = context["y_va"]
    keys_va = context["keys_va"]

    t0 = time.perf_counter()
    rf, res, _ = run_random_forest(
        X_tr, y_tr, X_va, y_va, keys_va,
        feature_group_name="G3", n_estimators=200,
        min_samples_leaf=50, max_samples=0.1, seed=42,
    )
    dur = time.perf_counter() - t0
    print(f"  RF: PR-AUC={res['pr_auc']:.6f}, R@10%={res['recall_top10']:.4f}, {dur:.1f}s")
    return {"pr_auc": res["pr_auc"], "recall_top10": res["recall_top10"],
            "brier": res["brier"], "seconds": round(dur, 1)}


def step5_secondary_label(winner_cfg: dict[str, Any]) -> dict[str, Any]:
    """Score winner on secondary label fire_next_48h (all rows)."""
    print("\n" + "=" * 70)
    print("STEP 5: Secondary Label (fire_next_48h, all rows)")
    print("=" * 70)
    assert_test_split_locked()

    label = "fire_next_48h"
    g3_cols = FEATURE_GROUPS["G3"]

    X_tr, y_tr, keys_tr = load_features(
        "train", label=label, feature_names=g3_cols,
        primary=False, features_path=FEATURES_FULL_PATH,
    )
    X_va, y_va, keys_va = load_features(
        "val", label=label, feature_names=g3_cols,
        primary=False, features_path=FEATURES_FULL_PATH,
    )
    print(f"  All-rows train: {len(X_tr):,} | val: {len(X_va):,}")

    winner_params = {k: v for k, v in winner_cfg.items() if k != "best_trees"}
    t0 = time.perf_counter()
    _, trees, scores = fit_xgb_two_stage(
        X_tr, y_tr, keys_tr["date"], X_va,
        params=winner_params, max_trees=1000, seed=42,
    )
    dur = time.perf_counter() - t0

    eval_df = keys_va.copy()
    eval_df[label] = y_va.values
    eval_df["score"] = scores
    res = evaluate(eval_df, label, "score", model="xgb_v1_winner_all_rows", split="val", set_name="all")
    record_result(res)
    print(f"  Secondary label (winner): PR-AUC={res['pr_auc']:.6f}, trees={trees} in {dur:.1f}s")
    return {"pr_auc": res["pr_auc"], "brier": res["brier"], "trees": trees}


def step6_winner(ablation_df: pd.DataFrame, rf_res: dict[str, Any]) -> dict[str, Any]:
    """Select Model v1 winner per Decision D18 rule."""
    print("\n" + "=" * 70)
    print("STEP 6: Model v1 Winner Selection (Decision D18)")
    print("=" * 70)

    candidates = []
    for _, row in ablation_df.iterrows():
        g = str(row["group"])
        p = str(row["params"])
        name = f"xgb_v1_{g}_{p}" if g != "climatology" else "climatology"
        candidates.append({
            "model": name,
            "group": g,
            "pr_auc": float(row["pr_auc"]),
            "recall_top10": float(row["recall_top10"]),
            "complexity": int(row["n_features"]),
        })

    candidates.append({
        "model": "rf_v1",
        "group": "RF_G3",
        "pr_auc": float(rf_res["pr_auc"]),
        "recall_top10": float(rf_res["recall_top10"]),
        "complexity": 21,
    })

    winner = select_winner(candidates)
    print(f"\n  Winner: {winner['model']} (PR-AUC={winner['pr_auc']:.6f}, R@10%={winner['recall_top10']:.4f})")
    print("\n  All candidates ranked:")
    for c in sorted(candidates, key=lambda x: -x["pr_auc"]):
        print(f"    {c['model']:28s}: PR-AUC={c['pr_auc']:.6f}, R@10%={c['recall_top10']:.4f}")

    return winner


def step7_feature_importance(winner_model: Any, X_va: pd.DataFrame, y_va: pd.Series) -> pd.DataFrame:
    """Compute native gain importance and validation permutation PR-AUC importance."""
    print("\n" + "=" * 70)
    print("STEP 7: Feature Importance (200k sample, 3 repeats)")
    print("=" * 70)
    g3_cols = FEATURE_GROUPS["G3"]

    imp_df = compute_feature_importance(
        winner_model, X_va, y_va, feature_names=g3_cols,
        n_sample=200_000, n_repeats=3, seed=42,
        output_path=DOCS_AUDIT_DIR / "full_run_feature_importance.csv",
    )
    print("\nTop 10 features by permutation importance:")
    print(imp_df.head(10).to_string(index=False))
    return imp_df


def step8_write_diagnostics(
    ablation_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    rf_res: dict[str, Any],
    sec_res: dict[str, Any],
    winner: dict[str, Any],
    imp_df: pd.DataFrame,
) -> None:
    """Write docs/audit/full_run_diagnostics.md."""
    print("\n" + "=" * 70)
    print("STEP 8: Writing Audit Diagnostics Markdown")
    print("=" * 70)

    lines = [
        "# Full-Cell Run Diagnostics (1,167 cells)",
        "",
        f"**Date:** {date.today()}",
        "**Grid Cells:** 1,167 (100% complete weather cache)",
        "**Status:** Full run complete, test split locked (2024-2025 untouched)",
        "",
        "## 1. Ablation Results (2023 Validation, Primary Set)",
        "",
        "| Group | Params | Features | PR-AUC | Recall@10% | Prec@5% | Brier | Trees | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in ablation_df.iterrows():
        lines.append(
            f"| {row['group']} | {row['params']} | {row['n_features']} | "
            f"{row['pr_auc']:.6f} | {row['recall_top10']:.4f} | "
            f"{row['precision_top5']:.4f} | {row['brier']:.6f} | {row['best_trees']} | {row['seconds']:.1f} |"
        )

    lines.extend([
        "",
        "## 2. Sensitivity Check (Fixed 150 Trees, No Early Stopping)",
        "",
        "| Group | Trees | PR-AUC | Recall@10% | Prec@5% | Brier |",
        "|---|---|---|---|---|---|",
    ])
    for _, row in fixed_df.iterrows():
        lines.append(
            f"| {row['group']} | {row['trees']} | {row['pr_auc']:.6f} | "
            f"{row['recall_top10']:.4f} | {row['precision_top5']:.4f} | {row['brier']:.6f} |"
        )

    lines.extend([
        "",
        "## 3. Block Bootstrap Uncertainty (1,000 Calendar-Week Resamples, Seed 42)",
        "",
        "| Contrast | Point Diff | 95% CI Lower | 95% CI Upper | Includes 0? | Verdict |",
        "|---|---|---|---|---|---|",
    ])
    for _, row in boot_df.iterrows():
        lines.append(
            f"| {row['contrast']} | {row['point_diff']:+.6f} | "
            f"{row['ci_lower_95']:+.6f} | {row['ci_upper_95']:+.6f} | "
            f"{'Yes' if row['includes_zero'] else 'No'} | {row['verdict']} |"
        )

    lines.extend([
        "",
        "## 4. Rolling-Origin Multi-Year Evaluation",
        "",
        "| Val Year | Group | Features | PR-AUC | Recall@10% | Prec@5% | Brier | Trees |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for _, row in rolling_df.iterrows():
        lines.append(
            f"| {row['val_year']} | {row['group']} | {row['n_features']} | {row['pr_auc']:.6f} | "
            f"{row['recall_top10']:.4f} | {row['precision_top5']:.4f} | {row['brier']:.6f} | {row['best_trees']} |"
        )

    lines.extend([
        "",
        "## 5. Benchmarks & Secondary Labels",
        "",
        f"- **Random Forest (G3, 200 trees):** PR-AUC: {rf_res['pr_auc']:.6f}, Recall@10%: {rf_res['recall_top10']:.4f}, Brier: {rf_res['brier']:.6f}",
        f"- **Secondary Label (`fire_next_48h`, all rows):** PR-AUC: {sec_res['pr_auc']:.6f}, Brier: {sec_res['brier']:.6f}",
        "",
        "## 6. Model v1 Winner Selection (Decision D18)",
        "",
        f"- **Winning Model:** `{winner['model']}`",
        f"- **Feature Group:** `{winner['group']}` ({winner['complexity']} features)",
        f"- **2023 Primary PR-AUC:** `{winner['pr_auc']:.6f}`",
        f"- **Recall@10%:** `{winner['recall_top10']:.4f}`",
        "",
        "## 7. Feature Importance (Permutation PR-AUC & Gain)",
        "",
        "| Rank | Feature | Perm Mean | Perm Std | Gain |",
        "|---|---|---|---|---|",
    ])
    for rank, (_, row) in enumerate(imp_df.head(15).iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['perm_importance_mean']:.6f} | "
            f"{row['perm_importance_std']:.6f} | {row['gain_importance']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "*All numbers traced to saved CSV files in `docs/audit/`.*",
    ])

    out_path = DOCS_AUDIT_DIR / "full_run_diagnostics.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved diagnostics markdown to {out_path}")


def phase_d() -> dict[str, Any]:
    """Execute complete Phase D pipeline."""
    t0 = time.perf_counter()

    tuned_cfg = load_tuned_config()
    print(f"Loaded tuned config: {json.dumps(tuned_cfg, indent=2)}")

    ablation_df, fixed_df, context = step1_ablation(tuned_cfg)
    boot_df, _ = step2_bootstrap(context["preds_df"])
    rolling_df = step3_rolling_origin()
    rf_res = step4_random_forest(context)
    sec_res = step5_secondary_label(tuned_cfg)
    winner = step6_winner(ablation_df, rf_res)

    winner_model = context["models"]["G3_tuned"][0]
    imp_df = step7_feature_importance(winner_model, context["X_va"], context["y_va"])

    step8_write_diagnostics(
        ablation_df, fixed_df, boot_df, rolling_df,
        rf_res, sec_res, winner, imp_df,
    )

    # STOP Gate: winner PR-AUC below climatology
    clim_prauc = float(ablation_df[ablation_df["group"] == "climatology"]["pr_auc"].iloc[0])
    if winner["pr_auc"] < clim_prauc:
        raise RuntimeError(
            f"STOP: Winner PR-AUC ({winner['pr_auc']:.6f}) below climatology ({clim_prauc:.6f})"
        )

    elapsed = time.perf_counter() - t0
    result = {
        "status": "PASS",
        "winner": winner["model"],
        "winner_pr_auc": round(winner["pr_auc"], 6),
        "winner_recall_top10": round(winner["recall_top10"], 4),
        "clim_pr_auc": round(clim_prauc, 6),
        "elapsed_s": round(elapsed, 1),
        "elapsed_min": round(elapsed / 60, 1),
    }
    print(f"\nPhase D completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return result


if __name__ == "__main__":
    result = phase_d()
    print(f"\n=== Phase D result: {json.dumps(result, indent=2)} ===")
