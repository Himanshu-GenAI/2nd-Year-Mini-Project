"""One-Shot Test Evaluation Script for Model v1.

Strictly enforces pre-registered anti-tampering guards:
1. Real mode requires explicit --i-have-approval flag.
2. Real mode refuses if any hash in configs/model_v1_frozen.yaml mismatches.
3. Real mode refuses if git working tree has uncommitted changes in src/, configs/, scripts/.
4. Real mode refuses if docs/audit/test_run_lock.json already exists.
5. In-process only assertion of ALLOW_TEST_SPLIT=1 (never persistent in shell/CI).
6. --rehearsal flag executes the exact same pipeline on 2023 validation split,
   writing to docs/audit/rehearsal/ and data/processed/rehearsal/, never writes
   lock file, and refuses if any 2024/2025 row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain.labels import GROUPS  # noqa: E402
from src.explain.shap_v0 import (  # noqa: E402
    compute_tree_shap_margins,
    extract_top_k_grouped_local_shap,
)
from src.models.baselines import fit_climatology, predict_climatology  # noqa: E402
from src.models.calibrate import load_model_v1_bundle  # noqa: E402
from src.models.dataset import load_split  # noqa: E402
from src.models.evaluate import precision_at_top_frac, recall_at_top_frac  # noqa: E402
from src.models.features_io import load_features  # noqa: E402

FROZEN_MODEL_YAML = PROJECT_ROOT / "configs" / "model_v1_frozen.yaml"
FROZEN_TIERS_YAML = PROJECT_ROOT / "configs" / "tiers_v1_frozen.yaml"
AUDIT_LOCK_JSON = PROJECT_ROOT / "docs" / "audit" / "test_run_lock.json"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_hashes(
    frozen_cfg_path: Path = FROZEN_MODEL_YAML,
    bundle_dir: Path | None = None,
    upstream_map: dict[str, Path] | None = None,
) -> dict[str, str]:
    """Verify SHA-256 hashes of all bundle and upstream artifacts against frozen record."""
    if not frozen_cfg_path.exists():
        raise FileNotFoundError(f"Frozen model configuration not found at {frozen_cfg_path}")

    with open(frozen_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    verified_hashes: dict[str, str] = {}
    b_dir = bundle_dir or (PROJECT_ROOT / "models" / "model_v1")
    for fname, expected_hash in cfg.get("bundle_files_sha256", {}).items():
        fpath = b_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Model bundle file missing: {fpath}")
        actual_hash = compute_sha256(fpath)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"INTEGRITY ERROR: Hash mismatch for bundle file {fname}!\n"
                f"Expected: {expected_hash}\nActual:   {actual_hash}"
            )
        verified_hashes[f"bundle/{fname}"] = actual_hash

    u_map = upstream_map or {
        "cells_final.json": PROJECT_ROOT / "data" / "processed" / "cells_final.json",
        "tiers_v1_preview.yaml": PROJECT_ROOT / "configs" / "tiers_v1_preview.yaml",
    }
    for fname, expected_hash in cfg.get("upstream_files_sha256", {}).items():
        fpath = u_map.get(fname)
        if not fpath or not fpath.exists():
            raise FileNotFoundError(f"Upstream file missing: {fname} at {fpath}")
        actual_hash = compute_sha256(fpath)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"INTEGRITY ERROR: Hash mismatch for upstream file {fname}!\n"
                f"Expected: {expected_hash}\nActual:   {actual_hash}"
            )
        verified_hashes[f"upstream/{fname}"] = actual_hash

    return verified_hashes


def check_git_cleanliness() -> None:
    """Refuse execution if git working tree has uncommitted modifications in code or configs."""
    cmd = ["git", "status", "--porcelain", "src", "configs", "scripts"]
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        # git binary not found; fail-safe refusal
        raise RuntimeError("Git command not found; cannot verify working tree cleanliness.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to check git status: {e.stderr}") from e

    out = proc.stdout.strip()
    if out:
        raise RuntimeError(
            "INTEGRITY REFUSAL: Git working directory has uncommitted changes in src/, configs/ or scripts/:\n" + out
        )


def check_lock_file(lock_path: Path = AUDIT_LOCK_JSON) -> None:
    """Refuse execution if the test run lock file already exists."""
    if lock_path.exists():
        raise RuntimeError(
            f"ANTI-TAMPERING REFUSAL: Test run lock file already exists at {lock_path}!\n"
            "Pre-registered protocol forbids repeated test evaluations."
        )


def format_verdict(
    prauc_model: float,
    prauc_clim: float,
    prauc_ci_lower: float,
    recall_model: float,
    recall_clim: float,
) -> tuple[bool, str]:
    """Format o1_verdict.md strictly containing ONLY the three conditions with unrounded values and PASS/FAIL."""
    rel_gain = (prauc_model / prauc_clim) - 1.0 if prauc_clim > 0 else float("-inf")
    cond_a_pass = rel_gain >= 0.15
    cond_b_pass = prauc_ci_lower > 0.0
    rec_diff = recall_model - recall_clim
    cond_c_pass = rec_diff >= 0.0

    overall_pass = cond_a_pass and cond_b_pass and cond_c_pass

    status_a = "PASS" if cond_a_pass else "FAIL"
    status_b = "PASS" if cond_b_pass else "FAIL"
    status_c = "PASS" if cond_c_pass else "FAIL"
    status_overall = "PASS" if overall_pass else "FAIL"

    lines = [
        f"Condition (a): {rel_gain} - {status_a}",
        f"Condition (b): {prauc_ci_lower} - {status_b}",
        f"Condition (c): {rec_diff} - {status_c}",
        f"Overall: {status_overall}",
    ]
    return overall_pass, "\n".join(lines) + "\n"


def compute_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute binary confusion matrix and classification rates at a specific threshold."""
    y_b = y_true.astype(bool)
    pred = y_score >= threshold

    tp = int(np.sum(pred & y_b))
    fp = int(np.sum(pred & ~y_b))
    fn = int(np.sum(~pred & y_b))
    tn = int(np.sum(~pred & ~y_b))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
    }


def compute_metrics_for_slice(
    df: pd.DataFrame,
    label: str,
    score_col: str,
    model_name: str,
    period: str,
    set_name: str,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Compute comprehensive metric suite for a specific temporal/label slice."""
    y_true = df[label].values.astype(int)
    scores = df[score_col].values.astype(float)

    n_rows = len(df)
    n_pos = int(np.sum(y_true))
    br = float(np.mean(y_true)) if n_rows > 0 else 0.0

    pr_val = float(average_precision_score(y_true, scores)) if n_pos > 0 else float("nan")
    pr_over_base = pr_val / br if (br > 0 and not math.isnan(pr_val)) else float("nan")

    rec_top10, dates_used = recall_at_top_frac(df, label, score_col, frac=0.10)
    prec_top5 = precision_at_top_frac(df, label, score_col, frac=0.05)
    brier_val = float(brier_score_loss(y_true, scores))

    metric_rows = [
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "rows",
            "value": n_rows,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "positives",
            "value": n_pos,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "base_rate",
            "value": br,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "pr_auc",
            "value": pr_val,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "pr_auc_over_base",
            "value": pr_over_base,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "recall_top10",
            "value": rec_top10,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "precision_top5",
            "value": prec_top5,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "brier",
            "value": brier_val,
        },
        {
            "period": period,
            "label": label,
            "set_name": set_name,
            "model": model_name,
            "metric": "dates_used",
            "value": dates_used,
        },
    ]

    for t_name, t_val in thresholds.items():
        cm = compute_confusion_matrix(y_true, scores, t_val)
        for cm_k, cm_v in cm.items():
            metric_rows.append(
                {
                    "period": period,
                    "label": label,
                    "set_name": set_name,
                    "model": model_name,
                    "metric": f"cm_{t_name}_{cm_k}",
                    "value": cm_v,
                }
            )

    return metric_rows


def run_week_block_bootstrap(
    df: pd.DataFrame,
    label: str,
    model_score_col: str,
    clim_score_col: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run 1,000 calendar-week block bootstrap for PR-AUC and recall@10% difference."""
    dates = pd.to_datetime(df["date"])
    weeks = dates.dt.to_period("W").astype(str).values
    unique_weeks = np.unique(weeks)
    n_unique_weeks = len(unique_weeks)

    week_indices = {w: np.where(weeks == w)[0] for w in unique_weeks}

    # Precompute per-day recalls for fast bootstrap aggregation
    day_rec_model: dict[Any, float] = {}
    day_rec_clim: dict[Any, float] = {}
    day_to_week: dict[Any, str] = {}

    for d_val, day_df in df.groupby("date"):
        pos_total = int(day_df[label].sum())
        if pos_total == 0:
            continue
        w_val = str(pd.to_datetime(d_val).to_period("W"))
        day_to_week[d_val] = w_val

        n_day = len(day_df)
        k = math.ceil(0.10 * n_day)

        # Model top 10%
        sorted_m = day_df.sort_values(by=[model_score_col, "cell_id"], ascending=[False, True])
        top_k_m = sorted_m.iloc[:k]
        day_rec_model[d_val] = float(top_k_m[label].sum()) / pos_total

        # Climatology top 10%
        sorted_c = day_df.sort_values(by=[clim_score_col, "cell_id"], ascending=[False, True])
        top_k_c = sorted_c.iloc[:k]
        day_rec_clim[d_val] = float(top_k_c[label].sum()) / pos_total

    valid_days = list(day_rec_model.keys())
    week_to_days: dict[str, list[Any]] = {w: [] for w in unique_weeks}
    for d_val in valid_days:
        week_to_days[day_to_week[d_val]].append(d_val)

    y_all = df[label].values.astype(int)
    s_m_all = df[model_score_col].values.astype(float)
    s_c_all = df[clim_score_col].values.astype(float)

    # Point diffs
    point_pr_m = float(average_precision_score(y_all, s_m_all))
    point_pr_c = float(average_precision_score(y_all, s_c_all))
    point_diff_pr = point_pr_m - point_pr_c

    point_rec_m = float(np.mean(list(day_rec_model.values()))) if day_rec_model else 0.0
    point_rec_c = float(np.mean(list(day_rec_clim.values()))) if day_rec_clim else 0.0
    point_diff_rec = point_rec_m - point_rec_c

    rng = np.random.default_rng(seed)
    pr_diffs = np.empty(n_boot, dtype=np.float64)
    rec_diffs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_w = rng.choice(unique_weeks, size=n_unique_weeks, replace=True)
        idx_b = np.concatenate([week_indices[w] for w in sampled_w])

        y_b = y_all[idx_b]
        if np.sum(y_b) == 0:
            pr_diffs[b] = 0.0
        else:
            pr_m_b = float(average_precision_score(y_b, s_m_all[idx_b]))
            pr_c_b = float(average_precision_score(y_b, s_c_all[idx_b]))
            pr_diffs[b] = pr_m_b - pr_c_b

        # Sampled days for recall
        sampled_days = [d for w in sampled_w for d in week_to_days[w]]
        if sampled_days:
            rec_m_b = float(np.mean([day_rec_model[d] for d in sampled_days]))
            rec_c_b = float(np.mean([day_rec_clim[d] for d in sampled_days]))
            rec_diffs[b] = rec_m_b - rec_c_b
        else:
            rec_diffs[b] = 0.0

    ci_low_pr = float(np.percentile(pr_diffs, 2.5))
    ci_high_pr = float(np.percentile(pr_diffs, 97.5))
    summary_pr = {
        "metric": "pr_auc",
        "point_diff": point_diff_pr,
        "boot_mean": float(np.mean(pr_diffs)),
        "boot_std": float(np.std(pr_diffs, ddof=1)),
        "ci_lower_95": ci_low_pr,
        "ci_upper_95": ci_high_pr,
        "includes_zero": bool(ci_low_pr <= 0.0 <= ci_high_pr),
    }

    ci_low_rec = float(np.percentile(rec_diffs, 2.5))
    ci_high_rec = float(np.percentile(rec_diffs, 97.5))
    summary_rec = {
        "metric": "recall_top10",
        "point_diff": point_diff_rec,
        "boot_mean": float(np.mean(rec_diffs)),
        "boot_std": float(np.std(rec_diffs, ddof=1)),
        "ci_lower_95": ci_low_rec,
        "ci_upper_95": ci_high_rec,
        "includes_zero": bool(ci_low_rec <= 0.0 <= ci_high_rec),
    }

    return summary_pr, summary_rec


def run_test_pipeline(rehearsal: bool = False, approval: bool = False) -> dict[str, Any]:
    """Execute complete test evaluation pipeline in either rehearsal or real mode."""
    t0 = time.perf_counter()

    if not rehearsal and not approval:
        raise RuntimeError(
            "INTEGRITY REFUSAL: Real test run requires explicit --i-have-approval flag.\n"
            "If this is a dry run rehearsal, use --rehearsal instead."
        )

    # 1. Hashes verification
    verified_hashes = verify_frozen_hashes()

    # 2. Real mode guards
    if not rehearsal:
        check_git_cleanliness()
        check_lock_file()
        # In-process assertion only
        os.environ["ALLOW_TEST_SPLIT"] = "1"
        target_split = "test"
        audit_dir = PROJECT_ROOT / "docs" / "audit"
        data_pred_dir = PROJECT_ROOT / "data" / "processed" / "predictions"
        data_shap_dir = PROJECT_ROOT / "data" / "processed" / "shap"
    else:
        target_split = "val"
        audit_dir = PROJECT_ROOT / "docs" / "audit" / "rehearsal"
        data_pred_dir = PROJECT_ROOT / "data" / "processed" / "rehearsal" / "predictions"
        data_shap_dir = PROJECT_ROOT / "data" / "processed" / "rehearsal" / "shap"

    audit_dir.mkdir(parents=True, exist_ok=True)
    data_pred_dir.mkdir(parents=True, exist_ok=True)
    data_shap_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load Model v1 and frozen tiers
    bundle = load_model_v1_bundle(verify_checksums=True)
    model = bundle["model"]
    feature_names = bundle["features"]
    model_version = bundle["version"]

    with open(FROZEN_TIERS_YAML, encoding="utf-8") as f:
        tiers_cfg = yaml.safe_load(f)

    p90 = float(tiers_cfg["thresholds"]["p90"])
    p97 = float(tiers_cfg["thresholds"]["p97"])
    p99_5 = float(tiers_cfg["thresholds"]["p99_5"])
    thresholds = {"p90": p90, "p97": p97, "p99_5": p99_5}

    # 4. Fit historical Climatology on train split (2015-2022) only
    train_df = load_split("train", label="fire_next_48h", primary=False)
    clim_model_new = fit_climatology(train_df, label="new_fire_next_48h", alpha=1.0)
    clim_model_fire = fit_climatology(train_df, label="fire_next_48h", alpha=1.0)

    # 5. Load target split features for all rows
    X_target, y_target_new, keys_target = load_features(
        target_split, label="new_fire_next_48h", feature_names=feature_names, primary=False
    )
    _, y_target_fire, _ = load_features(target_split, label="fire_next_48h", feature_names=feature_names, primary=False)

    # Temporal bounds verification
    target_years = pd.to_datetime(keys_target["date"]).dt.year.unique().tolist()
    if rehearsal:
        if any(y in (2024, 2025) for y in target_years):
            raise RuntimeError(f"REHEARSAL ERROR: Touched test years 2024/2025: {target_years}")
        if target_years != [2023]:
            raise RuntimeError(f"REHEARSAL ERROR: Expected only 2023 rows, found {target_years}")
    else:
        if any(y not in (2024, 2025) for y in target_years):
            raise RuntimeError(f"REAL MODE ERROR: Target years mismatch: expected [2024, 2025], got {target_years}")

    # 6. Inference
    raw_scores = model.predict_proba(X_target[feature_names])[:, 1].astype(np.float32)

    eval_df_clim = keys_target[["cell_id", "date"]].copy()
    clim_scores_new = predict_climatology(clim_model_new, eval_df_clim)
    clim_scores_fire = predict_climatology(clim_model_fire, eval_df_clim)

    burning_mask = keys_target["burning_prev3"].astype(bool).to_numpy()

    # 7. Apply tiers & build replay predictions table
    replay_scores = raw_scores.copy()
    replay_scores[burning_mask] = np.nan

    tier_assignments = np.empty(len(raw_scores), dtype=object)
    tier_assignments[burning_mask] = "Already burning"

    primary_idx = np.where(~burning_mask)[0]
    p_scores = raw_scores[primary_idx]
    p_tiers = np.where(
        p_scores >= p99_5,
        "Critical",
        np.where(
            p_scores >= p97,
            "High",
            np.where(p_scores >= p90, "Moderate", "Low"),
        ),
    )
    tier_assignments[primary_idx] = p_tiers

    category_type = pd.CategoricalDtype(
        categories=["Low", "Moderate", "High", "Critical", "Already burning"], ordered=False
    )
    replay_df = pd.DataFrame(
        {
            "cell_id": keys_target["cell_id"].astype(str),
            "date": pd.to_datetime(keys_target["date"]),
            "score": pd.Series(replay_scores, dtype="float32"),
            "tier": pd.Series(tier_assignments, dtype=category_type),
            "model_version": str(model_version),
            "split": str(target_split),
            "burning_prev3": keys_target["burning_prev3"].astype(bool),
        }
    )

    pred_replay_path = data_pred_dir / "model_v1_test_replay.parquet"
    replay_df.to_parquet(pred_replay_path, index=False)

    # 8. Compute local grouped TreeSHAP for High and Critical rows
    hc_mask = np.isin(tier_assignments, ["High", "Critical"])
    hc_indices = np.where(hc_mask)[0]
    shap_replay_path = data_shap_dir / "model_v1_test_replay.parquet"

    if len(hc_indices) > 0:
        X_hc = X_target.iloc[hc_indices][feature_names].reset_index(drop=True)
        keys_hc = keys_target.iloc[hc_indices][["cell_id", "date"]].reset_index(drop=True)

        shap_vals_hc, _ = compute_tree_shap_margins(model, X_hc)
        shap_grouped_df = extract_top_k_grouped_local_shap(
            keys_df=keys_hc,
            X_df=X_hc,
            shap_values=shap_vals_hc,
            feature_names=feature_names,
            model_version=str(model_version),
            k=5,
            group_mapping=GROUPS,
        )
        shap_grouped_df.to_parquet(shap_replay_path, index=False)
    else:
        # Empty placeholder if no High/Critical rows
        empty_shap_df = pd.DataFrame(
            columns=["cell_id", "date", "rank", "driver", "shap", "feature_detail", "model_version"]
        )
        empty_shap_df.to_parquet(shap_replay_path, index=False)

    # 9. Comprehensive Metric Evaluation
    dates_s = pd.to_datetime(keys_target["date"])
    years_s = dates_s.dt.year

    if rehearsal:
        periods = [("pooled", np.ones(len(keys_target), dtype=bool)), ("2023", years_s == 2023)]
    else:
        periods = [
            ("pooled", np.ones(len(keys_target), dtype=bool)),
            ("2024", years_s == 2024),
            ("2025", years_s == 2025),
        ]

    results_rows: list[dict[str, Any]] = []

    # Store pooled primary values for final verdict
    pooled_model_prauc = 0.0
    pooled_clim_prauc = 0.0
    pooled_model_rec10 = 0.0
    pooled_clim_rec10 = 0.0

    for period_name, period_mask in periods:
        # A. Primary Label (new_fire_next_48h, ~burning_prev3)
        mask_prim = period_mask & (~burning_mask)
        df_prim = pd.DataFrame(
            {
                "cell_id": keys_target["cell_id"][mask_prim],
                "date": keys_target["date"][mask_prim],
                "new_fire_next_48h": y_target_new[mask_prim],
                "score_model": raw_scores[mask_prim],
                "score_clim": clim_scores_new[mask_prim],
            }
        )

        rows_m = compute_metrics_for_slice(
            df_prim, "new_fire_next_48h", "score_model", "model_v1", period_name, "primary", thresholds
        )
        rows_c = compute_metrics_for_slice(
            df_prim, "new_fire_next_48h", "score_clim", "climatology", period_name, "primary", thresholds
        )
        results_rows.extend(rows_m)
        results_rows.extend(rows_c)

        if period_name == "pooled":
            for r in rows_m:
                if r["metric"] == "pr_auc":
                    pooled_model_prauc = float(r["value"])
                elif r["metric"] == "recall_top10":
                    pooled_model_rec10 = float(r["value"])
            for r in rows_c:
                if r["metric"] == "pr_auc":
                    pooled_clim_prauc = float(r["value"])
                elif r["metric"] == "recall_top10":
                    pooled_clim_rec10 = float(r["value"])

        # B. Secondary Label (fire_next_48h, all rows)
        df_all = pd.DataFrame(
            {
                "cell_id": keys_target["cell_id"][period_mask],
                "date": keys_target["date"][period_mask],
                "fire_next_48h": y_target_fire[period_mask],
                "score_model": raw_scores[period_mask],
                "score_clim": clim_scores_fire[period_mask],
            }
        )

        rows_m_all = compute_metrics_for_slice(
            df_all, "fire_next_48h", "score_model", "model_v1", period_name, "all", thresholds
        )
        rows_c_all = compute_metrics_for_slice(
            df_all, "fire_next_48h", "score_clim", "climatology", period_name, "all", thresholds
        )
        results_rows.extend(rows_m_all)
        results_rows.extend(rows_c_all)

    results_csv_path = audit_dir / "test_results.csv"
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_csv_path, index=False)

    # 10. Block Bootstrap by Calendar Week (1,000 resamples, seed 42)
    bootstrap_summary_rows: list[dict[str, Any]] = []
    pooled_ci_lower_prauc = 0.0

    for period_name, period_mask in periods:
        mask_prim = period_mask & (~burning_mask)
        df_boot = pd.DataFrame(
            {
                "cell_id": keys_target["cell_id"][mask_prim],
                "date": keys_target["date"][mask_prim],
                "new_fire_next_48h": y_target_new[mask_prim],
                "score_model": raw_scores[mask_prim],
                "score_clim": clim_scores_new[mask_prim],
            }
        )

        sum_pr, sum_rec = run_week_block_bootstrap(
            df_boot,
            label="new_fire_next_48h",
            model_score_col="score_model",
            clim_score_col="score_clim",
            n_boot=1000,
            seed=42,
        )

        bootstrap_summary_rows.append(
            {
                "period": period_name,
                "contrast": "model_v1 - climatology",
                **sum_pr,
            }
        )
        bootstrap_summary_rows.append(
            {
                "period": period_name,
                "contrast": "model_v1 - climatology",
                **sum_rec,
            }
        )

        if period_name == "pooled":
            pooled_ci_lower_prauc = sum_pr["ci_lower_95"]

    bootstrap_csv_path = audit_dir / "test_bootstrap.csv"
    boot_df = pd.DataFrame(bootstrap_summary_rows)
    boot_df.to_csv(bootstrap_csv_path, index=False)

    # 11. Verdict Evaluation
    overall_pass, verdict_text = format_verdict(
        prauc_model=pooled_model_prauc,
        prauc_clim=pooled_clim_prauc,
        prauc_ci_lower=pooled_ci_lower_prauc,
        recall_model=pooled_model_rec10,
        recall_clim=pooled_clim_rec10,
    )

    verdict_md_path = audit_dir / "o1_verdict.md"
    verdict_md_path.write_text(verdict_text, encoding="utf-8")

    # 12. Write lock file in real mode
    if not rehearsal:
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
        lock_data = {
            "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_commit,
            "model_version": str(model_version),
            "status": "COMPLETED",
            "verdict": "PASS" if overall_pass else "FAIL",
            "script": "scripts/run_test_once.py",
            "verified_hashes": verified_hashes,
        }
        with open(AUDIT_LOCK_JSON, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

    runtime_s = time.perf_counter() - t0

    # Summary report
    pred_bytes = pred_replay_path.stat().st_size if pred_replay_path.exists() else 0
    shap_bytes = shap_replay_path.stat().st_size if shap_replay_path.exists() else 0

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "mode": "rehearsal" if rehearsal else "real",
        "runtime_s": runtime_s,
        "results_rows": len(results_df),
        "replay_pred_rows": len(replay_df),
        "replay_pred_size_bytes": pred_bytes,
        "replay_shap_rows": len(shap_grouped_df) if len(hc_indices) > 0 else 0,
        "replay_shap_size_bytes": shap_bytes,
        "verdict_text": verdict_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot test execution script for Model v1.")
    parser.add_argument("--rehearsal", action="store_true", help="Execute on 2023 validation set as a rehearsal.")
    parser.add_argument("--i-have-approval", action="store_true", help="Explicit approval flag required for real test.")
    args = parser.parse_args()

    res = run_test_pipeline(rehearsal=args.rehearsal, approval=args.i_have_approval)

    print("==================================================")
    print(f"Model v1 Test Pipeline Finished ({res['mode'].upper()} MODE)")
    print("==================================================")
    print(f"Runtime:             {res['runtime_s']:.2f} s")
    print(f"Results Rows:        {res['results_rows']}")
    print(f"Replay Pred Rows:    {res['replay_pred_rows']} ({res['replay_pred_size_bytes'] / (1024 * 1024):.2f} MB)")
    print(f"Replay SHAP Rows:    {res['replay_shap_rows']} ({res['replay_shap_size_bytes'] / (1024 * 1024):.2f} MB)")
    print("--------------------------------------------------")
    print("VERDICT:")
    print(res["verdict_text"].strip())
    print("==================================================")


if __name__ == "__main__":
    main()
