"""Development predictions and tier preview generation for Model v0.

Loads (or trains and verifies) xgb_v0_plain (393 trees, 2015-2022 primary train),
scores 2023 validation primary-set rows, computes P90/P97/P99.5 tier thresholds,
and exports model_v0_dev.parquet and configs/tiers_v0_preview.yaml.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.evaluate import pr_auc
from src.models.features_io import load_features
from src.models.train_v0 import V0_FEATURES

MODEL_V0_DIR = PROJECT_ROOT / "models" / "xgb_v0_plain"
MODEL_V0_PATH = MODEL_V0_DIR / "model.json"
PREDICTIONS_V0_PATH = PROCESSED_DIR / "predictions" / "model_v0_dev.parquet"
TIERS_PREVIEW_V0_PATH = PROJECT_ROOT / "configs" / "tiers_v0_preview.yaml"

EXPECTED_VAL_PR_AUC = 0.2478
PR_AUC_TOLERANCE = 0.0005
N_TREES_V0_PLAIN = 393


def compute_tier_thresholds(scores: np.ndarray) -> dict[str, float]:
    """Compute P90, P97, and P99.5 score thresholds."""
    p90, p97, p995 = np.percentile(scores, [90.0, 97.0, 99.5])
    return {
        "p90": float(p90),
        "p97": float(p97),
        "p99_5": float(p995),
    }


def assign_tiers(scores: np.ndarray, thresholds: dict[str, float]) -> pd.Series:
    """Map continuous probabilities to discrete risk tiers: Low, Moderate, High, Critical."""
    p90 = thresholds["p90"]
    p97 = thresholds["p97"]
    p995 = thresholds["p99_5"]

    conditions = [
        scores < p90,
        (scores >= p90) & (scores < p97),
        (scores >= p97) & (scores < p995),
        scores >= p995,
    ]
    choices = ["Low", "Moderate", "High", "Critical"]
    tiers = np.select(conditions, choices, default="Low")
    return pd.Series(tiers, dtype="category")


def load_or_train_v0_plain(model_path: Path = MODEL_V0_PATH) -> xgb.XGBClassifier:
    """Load saved xgb_v0_plain model or train on 2015-2022 primary rows (393 trees)."""
    model = xgb.XGBClassifier(
        n_estimators=N_TREES_V0_PLAIN,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
    )
    if model_path.exists():
        model.load_model(model_path)
        return model

    X_tr, y_tr, _ = load_features("train", label="new_fire_next_48h", feature_names=V0_FEATURES, primary=True)
    model.fit(X_tr, y_tr, verbose=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    return model


def generate_v0_dev_predictions(
    run_date: date | None = None,
    model_path: Path = MODEL_V0_PATH,
    output_path: Path = PREDICTIONS_V0_PATH,
    tiers_path: Path = TIERS_PREVIEW_V0_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score 2023 validation primary-set rows and export predictions and tier preview config."""
    today = run_date or date.today()
    model_version = f"v0_plain_{today.strftime('%Y%m%d')}"

    model = load_or_train_v0_plain(model_path=model_path)

    # Strictly load ONLY validation split (2023) primary set — never train or test
    X_va, y_va, keys_va = load_features("val", label="new_fire_next_48h", feature_names=V0_FEATURES, primary=True)

    scores = model.predict_proba(X_va)[:, 1].astype(np.float32)

    # Step 1 verification: 2023 primary-set PR-AUC must match 0.2478 within 0.0005
    eval_df = pd.DataFrame({"new_fire_next_48h": y_va.values, "score": scores})
    val_prauc = pr_auc(eval_df, "new_fire_next_48h", "score")
    diff = abs(val_prauc - EXPECTED_VAL_PR_AUC)
    print(f"Verified 2023 primary PR-AUC: {val_prauc:.6f} (expected {EXPECTED_VAL_PR_AUC:.4f}, diff={diff:.6f})")
    if diff > PR_AUC_TOLERANCE:
        raise RuntimeError(
            f"STOP: 2023 primary PR-AUC {val_prauc:.6f} differs from {EXPECTED_VAL_PR_AUC} "
            f"by {diff:.6f} > {PR_AUC_TOLERANCE}."
        )

    # Step 4: compute P90, P97, P99.5 on 2023 primary-set scores
    thresholds = compute_tier_thresholds(scores)
    tiers = assign_tiers(scores, thresholds)

    pred_df = pd.DataFrame(
        {
            "cell_id": keys_va["cell_id"].astype(str),
            "date": pd.to_datetime(keys_va["date"]),
            "score": scores,
            "tier": tiers,
            "model_version": model_version,
            "split": "val",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(output_path, index=False)

    # Tier shares
    counts = pred_df["tier"].value_counts()
    n_total = len(pred_df)
    shares = {t: round(float(counts.get(t, 0)) / n_total * 100.0, 4) for t in ["Low", "Moderate", "High", "Critical"]}

    tiers_config = {
        "model_version": model_version,
        "date": str(today),
        "label": "new_fire_next_48h",
        "eval_set": "val_2023_primary",
        "rows_scored": int(n_total),
        "val_pr_auc": round(float(val_prauc), 6),
        "thresholds": {
            "p90": round(thresholds["p90"], 6),
            "p97": round(thresholds["p97"], 6),
            "p99_5": round(thresholds["p99_5"], 6),
        },
        "tier_rules": {
            "Low": "score < p90",
            "Moderate": "p90 <= score < p97",
            "High": "p97 <= score < p99_5",
            "Critical": "score >= p99_5",
        },
        "tier_shares_pct": shares,
        "status": "preview_only_not_frozen",
    }

    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tiers_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tiers_config, f, sort_keys=False)

    print(f"Saved predictions ({n_total:,} rows) to {output_path}")
    print(f"Saved tier preview config to {tiers_path}")
    print(f"Thresholds: P90={thresholds['p90']:.6f}, P97={thresholds['p97']:.6f}, P99.5={thresholds['p99_5']:.6f}")
    print(f"Tier shares (%): {shares}")

    return pred_df, tiers_config


if __name__ == "__main__":
    generate_v0_dev_predictions()

