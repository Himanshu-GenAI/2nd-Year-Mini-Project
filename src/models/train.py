"""Wildfire Risk Model Training and Temporal Validation Pipeline.

Trains tree-based classifiers (XGBoost / Random Forest) using strictly temporal splits:
- Training:   < 2021-01-01
- Validation: 2021-01-01 to 2021-12-31 (used for calibration & threshold tuning)
- Test:       >= 2022-01-01 (strict out-of-time evaluation)
Saves calibrated model artifact to models/saved_models/.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.config import (
    CALIBRATION_METHOD,
    FEATURE_COLUMNS,
    SAVED_MODELS_DIR,
    TARGET_COLUMN,
    TEST_START_DATE,
    TRAIN_END_DATE,
    VAL_END_DATE,
    VAL_START_DATE,
    XGB_PARAMS,
)
from src.models.baselines import (
    PersistenceBaseline,
    evaluate_model_predictions,
)
from src.models.calibrate import CalibratedWildfireModel, compute_calibration_metrics


def split_temporal_dataset(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = TRAIN_END_DATE,
    val_start: str = VAL_START_DATE,
    val_end: str = VAL_END_DATE,
    test_start: str = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into strict chronological non-overlapping splits.

    Ensures zero temporal leakage:
    train.max(date) < val.min(date) and val.max(date) < test.min(date).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    train_df = df[df[date_col] <= train_end].copy()
    val_df = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)].copy()
    test_df = df[df[date_col] >= test_start].copy()

    return train_df, val_df, test_df


def train_wildfire_model(
    df: pd.DataFrame,
    feature_cols: list | None = None,
    target_col: str = TARGET_COLUMN,
    calibrate: bool = True,
    save_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute end-to-end training, validation, and calibration pipeline.

    Args:
        df: Processed feature dataset containing feature_cols, target_col, date, and grid_id.
        feature_cols: List of predictor column names.
        target_col: Binary target label name.
        calibrate: Whether to calibrate predicted probabilities on validation set.
        save_path: Path to save trained model bundle.

    Returns:
        Tuple of (model_pipeline, evaluation_metrics_dict).
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    train_df, val_df, test_df = split_temporal_dataset(df)

    # In case splits are empty (e.g. during small test runs), use proportional temporal split
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        df_sorted = df.sort_values("date")
        n = len(df_sorted)
        train_df = df_sorted.iloc[: int(0.6 * n)].copy()
        val_df = df_sorted.iloc[int(0.6 * n) : int(0.8 * n)].copy()
        test_df = df_sorted.iloc[int(0.8 * n) :].copy()

    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

    # Calculate dynamic scale_pos_weight based on class ratio
    neg_count = np.sum(y_train == 0)
    pos_count = max(1, np.sum(y_train == 1))
    scale_weight = float(neg_count / pos_count)

    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = min(scale_weight, 15.0)

    base_model = XGBClassifier(**params)
    base_model.fit(X_train, y_train)

    # Calibration on distinct temporal validation partition
    if calibrate and len(np.unique(y_val)) > 1:
        calibrator = CalibratedWildfireModel(base_model, method=CALIBRATION_METHOD)
        calibrator.fit_calibration(X_val, y_val)
        final_model = calibrator
    else:
        final_model = base_model

    # Evaluation on holdout test set
    y_test_probs = (
        final_model.predict_proba(X_test)[:, 1]
        if hasattr(final_model, "predict_proba")
        else base_model.predict_proba(X_test)[:, 1]
    )
    test_metrics = evaluate_model_predictions(y_test, y_test_probs)
    cal_metrics = compute_calibration_metrics(y_test, y_test_probs)

    # Baseline comparison (Persistence baseline)
    persistence = PersistenceBaseline()
    if "fire_count_lag_1d" in test_df.columns:
        p_probs = persistence.predict_proba(test_df)[:, 1]
        p_metrics = evaluate_model_predictions(y_test, p_probs)
    else:
        p_metrics = {}

    summary = {
        "model_type": "XGBoost",
        "calibrated": calibrate,
        "calibration_method": CALIBRATION_METHOD,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "test_metrics": test_metrics,
        "calibration_metrics": cal_metrics,
        "baseline_persistence": p_metrics,
        "features": feature_cols,
    }

    # Save model artifacts
    if save_path is None:
        save_path = SAVED_MODELS_DIR / "wildfire_model.joblib"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "features": feature_cols,
        "summary": summary,
    }
    joblib.dump(bundle, save_path)

    # Also save metadata json
    meta_path = save_path.parent / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    return final_model, summary


def evaluate_pipeline(model_path: Path, test_df: pd.DataFrame) -> dict[str, Any]:
    """Load model artifact and evaluate on arbitrary test panel."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    X = test_df[features].values
    y = test_df[TARGET_COLUMN].values
    probs = model.predict_proba(X)[:, 1]

    return evaluate_model_predictions(y, probs)
