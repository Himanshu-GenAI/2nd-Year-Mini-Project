"""Post-hoc probability calibration, quarterly blocked CV, breakdowns, and Model v1 bundle I/O.

Replaces the Week 1 placeholder template. Uses IsotonicRegression directly
(avoiding removed CalibratedClassifierCV(cv="prefit") in scikit-learn >=1.6),
evaluates calibration strictly with 4-quarter blocked CV on 2023 validation data,
and packages portable Model v1 artifacts with SHA-256 checksums.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.isotonic import IsotonicRegression

from src.config import DOCS_AUDIT_DIR, PROJECT_ROOT
from src.models.dev_predictions import assign_tiers, compute_tier_thresholds
from src.models.evaluate import brier, plot_reliability, pr_auc, recall_at_top_frac, reliability_table
from src.models.train_v0 import prior_correct
from src.models.train_v1 import assert_test_split_locked

MODEL_V1_BUNDLE_DIR = PROJECT_ROOT / "models" / "model_v1"
BREAKDOWN_CSV_PATH = DOCS_AUDIT_DIR / "model_v1_breakdown.csv"
RELIABILITY_PNG_PATH = DOCS_AUDIT_DIR / "model_v1_reliability.png"
TIERS_PREVIEW_V1_PATH = PROJECT_ROOT / "configs" / "tiers_preview.yaml"

SEASON_MAP = {
    12: "Winter (Dec-Feb)",
    1: "Winter (Dec-Feb)",
    2: "Winter (Dec-Feb)",
    3: "Pre-monsoon (Mar-May)",
    4: "Pre-monsoon (Mar-May)",
    5: "Pre-monsoon (Mar-May)",
    6: "Monsoon (Jun-Sep)",
    7: "Monsoon (Jun-Sep)",
    8: "Monsoon (Jun-Sep)",
    9: "Monsoon (Jun-Sep)",
    10: "Post-monsoon (Oct-Nov)",
    11: "Post-monsoon (Oct-Nov)",
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit monotone isotonic regression mapping scores to probabilities in [0, 1]."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return iso


def predict_isotonic(calibrator: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    """Apply fitted isotonic calibrator and guarantee outputs in [0, 1]."""
    preds = calibrator.predict(np.asarray(scores, dtype=np.float64))
    return np.clip(preds, 0.0, 1.0)


def blocked_quarterly_cv(
    df_val: pd.DataFrame,
    label: str,
    score_col: str,
    date_col: str = "date",
    min_pos_for_judge: int = 100,
) -> dict[str, Any]:
    """Run 4-quarter temporal blocked CV on validation rows to test isotonic calibration.

    Trains isotonic regression on 3 quarters and evaluates out-of-fold on the held-out
    quarter. Keeps post-hoc calibration only if overall CV Brier < raw Brier.
    """
    work = df_val.copy().reset_index(drop=True)
    quarters = pd.to_datetime(work[date_col]).dt.quarter.values
    raw_scores = work[score_col].values.astype(np.float64)
    y_true = work[label].values.astype(np.int64)

    oof_cal = np.empty(len(work), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []

    for q in [1, 2, 3, 4]:
        val_mask = quarters == q
        tr_mask = ~val_mask
        n_q = int(val_mask.sum())
        if n_q == 0 or int(tr_mask.sum()) == 0:
            raise ValueError(f"Quarter Q{q} or its complement is empty in blocked_quarterly_cv.")

        iso = fit_isotonic(raw_scores[tr_mask], y_true[tr_mask])
        oof_cal[val_mask] = predict_isotonic(iso, raw_scores[val_mask])

        q_df = pd.DataFrame(
            {
                "cell_id": work.loc[val_mask, "cell_id"].values if "cell_id" in work.columns else np.arange(n_q),
                date_col: work.loc[val_mask, date_col].values,
                label: y_true[val_mask],
                "raw": raw_scores[val_mask],
                "cal": oof_cal[val_mask],
            }
        )
        n_pos = int(q_df[label].sum())
        note = "too few to judge" if n_pos < min_pos_for_judge else "ok"
        fold_rows.append(
            {
                "quarter": f"Q{q}",
                "rows": n_q,
                "positives": n_pos,
                "raw_brier": round(brier(q_df, label, "raw"), 6),
                "cal_brier": round(brier(q_df, label, "cal"), 6),
                "raw_pr_auc": round(pr_auc(q_df, label, "raw"), 6),
                "cal_pr_auc": round(pr_auc(q_df, label, "cal"), 6),
                "note": note,
            }
        )

    oof_df = pd.DataFrame({label: y_true, "raw": raw_scores, "cal": oof_cal})
    raw_brier_cv = float(brier(oof_df, label, "raw"))
    cal_brier_cv = float(brier(oof_df, label, "cal"))
    use_calibrator = bool(cal_brier_cv < raw_brier_cv)

    final_calibrator = fit_isotonic(raw_scores, y_true) if use_calibrator else None
    return {
        "use_calibrator": use_calibrator,
        "calibrator": final_calibrator,
        "raw_brier_cv": round(raw_brier_cv, 6),
        "cal_brier_cv": round(cal_brier_cv, 6),
        "folds_df": pd.DataFrame(fold_rows),
        "oof_scores": oof_cal,
    }


def _slice_metrics(
    sub: pd.DataFrame,
    dimension: str,
    bucket: str,
    label: str,
    score_col: str,
    min_pos_for_judge: int = 100,
) -> dict[str, Any]:
    n_rows = len(sub)
    n_pos = int(sub[label].sum())
    base_r = float(sub[label].mean()) if n_rows > 0 else float("nan")
    prauc = pr_auc(sub, label, score_col) if n_pos > 0 and n_pos < n_rows else float("nan")
    rec10, _ = recall_at_top_frac(sub, label, score_col, frac=0.10) if n_pos > 0 else (float("nan"), 0)
    brier_val = brier(sub, label, score_col) if n_rows > 0 else float("nan")
    note = "too few to judge" if n_pos < min_pos_for_judge else "ok"
    return {
        "dimension": dimension,
        "bucket": bucket,
        "rows": n_rows,
        "positives": n_pos,
        "base_rate": round(base_r, 6),
        "pr_auc": round(prauc, 6) if not np.isnan(prauc) else float("nan"),
        "recall_top10": round(rec10, 4) if not np.isnan(rec10) else float("nan"),
        "brier": round(brier_val, 6) if not np.isnan(brier_val) else float("nan"),
        "note": note,
    }


def compute_tiers_and_breakdowns(
    df_val_primary: pd.DataFrame,
    score_col: str,
    label: str = "new_fire_next_48h",
    df_val_all: pd.DataFrame | None = None,
    grid_df: pd.DataFrame | None = None,
    model_version: str = "v1",
    partial: bool = False,
    n_cells: int | None = None,
    min_pos_for_judge: int = 100,
    tiers_path: Path = TIERS_PREVIEW_V1_PATH,
    breakdown_path: Path = BREAKDOWN_CSV_PATH,
    reliability_path: Path = RELIABILITY_PNG_PATH,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute Model v1 tier thresholds, reliability diagram, and slice breakdowns."""
    work = df_val_primary.copy()
    thresholds = compute_tier_thresholds(work[score_col].values)
    work["tier"] = assign_tiers(work[score_col].values, thresholds)

    tier_shares = {
        str(k): round(float(v), 6)
        for k, v in work["tier"].value_counts(normalize=True).items()
    }
    tiers_payload = {
        "model_version": model_version,
        "date": str(date.today()),
        "split": "val",
        "set_name": f"partial: {n_cells} cells (primary)" if partial and n_cells is not None else "primary",
        "thresholds": thresholds,
        "tier_shares": tier_shares,
    }
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tiers_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tiers_payload, f, sort_keys=False)

    rel_tbl = reliability_table(work, label, score_col, bins=10)
    plot_reliability(rel_tbl, reliability_path)

    # Breakdown rows: seasons, top 10 states by positives, previous-activity groups
    rows: list[dict[str, Any]] = []
    work["season"] = pd.to_datetime(work["date"]).dt.month.map(SEASON_MAP)
    for season in ["Winter (Dec-Feb)", "Pre-monsoon (Mar-May)", "Monsoon (Jun-Sep)", "Post-monsoon (Oct-Nov)"]:
        sub = work[work["season"] == season]
        if len(sub) > 0:
            rows.append(_slice_metrics(sub, "season", season, label, score_col, min_pos_for_judge=min_pos_for_judge))

    if grid_df is not None and "state" not in work.columns and "state" in grid_df.columns:
        work = work.merge(grid_df[["cell_id", "state"]].drop_duplicates(), on="cell_id", how="left")

    if "state" in work.columns:
        state_pos = work.groupby("state", observed=True)[label].sum().sort_values(ascending=False)
        for st in state_pos.head(10).index:
            sub = work[work["state"] == st]
            rows.append(_slice_metrics(sub, "state", str(st), label, score_col, min_pos_for_judge=min_pos_for_judge))

    act_df = df_val_all if df_val_all is not None else work
    if "burning_prev3" in act_df.columns:
        for flag in [False, True]:
            sub = act_df[act_df["burning_prev3"].astype(bool) == flag]
            if len(sub) > 0:
                bucket_name = "burning_prev3=True" if flag else "burning_prev3=False"
                rows.append(_slice_metrics(sub, "previous_activity", bucket_name, label, score_col, min_pos_for_judge=min_pos_for_judge))

    breakdown_df = pd.DataFrame(rows)
    breakdown_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown_df.to_csv(breakdown_path, index=False)
    return thresholds, breakdown_df


def save_model_v1_bundle(
    model: Any,
    feature_names: list[str],
    fit_dates: dict[str, Any],
    params: dict[str, Any],
    version: str,
    calibrator: IsotonicRegression | None = None,
    bundle_dir: Path = MODEL_V1_BUNDLE_DIR,
) -> dict[str, str]:
    """Save portable Model v1 bundle with optional calibrator and SHA-256 checksums."""
    assert_test_split_locked()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: list[Path] = []
    if isinstance(model, xgb.XGBClassifier):
        model_path = bundle_dir / "model.json"
        model.save_model(model_path)
        model_file = "model.json"
        model_kind = "xgboost"
    else:
        model_path = bundle_dir / "model.joblib"
        joblib.dump(model, model_path)
        model_file = "model.joblib"
        model_kind = "sklearn"
    artifact_paths.append(model_path)

    cal_file: str | None = None
    if calibrator is not None:
        cal_path = bundle_dir / "calibrator.joblib"
        joblib.dump(calibrator, cal_path)
        cal_file = "calibrator.joblib"
        artifact_paths.append(cal_path)

    features_path = bundle_dir / "features.json"
    features_path.write_text(json.dumps(list(feature_names), indent=2), encoding="utf-8")
    artifact_paths.append(features_path)

    fit_dates_path = bundle_dir / "fit_dates.json"
    fit_dates_path.write_text(json.dumps(fit_dates, indent=2), encoding="utf-8")
    artifact_paths.append(fit_dates_path)

    params_payload = {
        "model_kind": model_kind,
        "model_file": model_file,
        "calibrator_file": cal_file,
        "calibrated": calibrator is not None,
        "prior_beta": float(params.get("prior_beta", 1.0)),
        "hyperparams": params,
    }
    params_path = bundle_dir / "params.json"
    params_path.write_text(json.dumps(params_payload, indent=2), encoding="utf-8")
    artifact_paths.append(params_path)

    version_path = bundle_dir / "version.txt"
    version_path.write_text(f"{version}\n", encoding="utf-8")
    artifact_paths.append(version_path)

    checksums = {p.name: sha256_file(p) for p in artifact_paths}
    checksum_path = bundle_dir / "checksums.json"
    checksum_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return checksums


def load_model_v1_bundle(bundle_dir: Path = MODEL_V1_BUNDLE_DIR, verify_checksums: bool = True) -> dict[str, Any]:
    """Load saved Model v1 bundle and verify artifact SHA-256 checksums."""
    checksum_path = bundle_dir / "checksums.json"
    if verify_checksums and checksum_path.exists():
        expected = json.loads(checksum_path.read_text(encoding="utf-8"))
        for fname, digest in expected.items():
            target = bundle_dir / fname
            if not target.exists():
                raise FileNotFoundError(f"Missing bundle file {target}")
            actual = sha256_file(target)
            if actual != digest:
                raise ValueError(f"Checksum mismatch for {fname}: expected {digest}, got {actual}")

    features = json.loads((bundle_dir / "features.json").read_text(encoding="utf-8"))
    fit_dates = json.loads((bundle_dir / "fit_dates.json").read_text(encoding="utf-8"))
    params_payload = json.loads((bundle_dir / "params.json").read_text(encoding="utf-8"))
    version = (bundle_dir / "version.txt").read_text(encoding="utf-8").strip()

    model_file = params_payload["model_file"]
    if params_payload["model_kind"] == "xgboost":
        model: Any = xgb.XGBClassifier()
        model.load_model(bundle_dir / model_file)
    else:
        model = joblib.load(bundle_dir / model_file)

    calibrator = None
    if params_payload.get("calibrator_file"):
        calibrator = joblib.load(bundle_dir / params_payload["calibrator_file"])

    return {
        "model": model,
        "calibrator": calibrator,
        "features": features,
        "fit_dates": fit_dates,
        "params": params_payload,
        "version": version,
    }


def predict_model_v1(bundle: dict[str, Any], X_df: pd.DataFrame) -> np.ndarray:
    """Run inference using loaded Model v1 bundle (strict column ordering + optional calibration)."""
    cols = bundle["features"]
    missing = [c for c in cols if c not in X_df.columns]
    if missing:
        raise ValueError(f"Missing required features for Model v1 inference: {missing}")

    raw_probs = bundle["model"].predict_proba(X_df[cols])[:, 1]
    beta = float(bundle["params"].get("prior_beta", 1.0))
    probs = np.asarray(prior_correct(raw_probs, beta=beta), dtype=np.float64)
    if bundle.get("calibrator") is not None:
        probs = predict_isotonic(bundle["calibrator"], probs)
    return np.clip(probs, 0.0, 1.0)
