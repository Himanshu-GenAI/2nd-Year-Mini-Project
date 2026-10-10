"""SHAP Explainability and Feature Attribution Utilities.

Computes local instance-level Shapley values and global feature importance rankings
for tree-based wildfire hazard models. Formats explanations for API and UI consumption.
"""

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _extract_base_model(model: Any) -> Any:
    """Extract underlying tree model if wrapped in CalibratedWildfireModel or CalibratedClassifierCV."""
    if hasattr(model, "calibrated_model") and model.calibrated_model is not None:
        model = model.calibrated_model

    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        return model.calibrated_classifiers_[0].estimator

    if hasattr(model, "base_estimator"):
        return model.base_estimator

    if hasattr(model, "estimator"):
        return model.estimator

    return model


def explain_prediction(
    model: Any,
    feature_names: list[str],
    instance_values: np.ndarray | pd.DataFrame | list[float],
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute local SHAP explanation for a single prediction.

    Args:
        model: Trained model (base or calibrated).
        feature_names: List of feature names.
        instance_values: Feature values for the grid cell under evaluation.
        top_k: Number of top driving factors to return.

    Returns:
        Dictionary with:
        - top_risk_drivers: List of features increasing fire risk (positive SHAP).
        - top_mitigating_factors: List of features reducing fire risk (negative SHAP).
        - all_contributions: Full dictionary of feature -> attribution value.
    """
    raw_model = _extract_base_model(model)

    if isinstance(instance_values, pd.DataFrame):
        X = instance_values[feature_names].values
    else:
        X = np.asarray(instance_values, dtype=float).reshape(1, -1)

    contributions: dict[str, float] = {}

    if HAS_SHAP and hasattr(raw_model, "predict"):
        try:
            explainer = shap.TreeExplainer(raw_model)
            shap_vals = explainer.shap_values(X)

            # Handle binary classification shapes (list of arrays or 2D/3D array)
            if isinstance(shap_vals, list):
                # Class 1 (fire presence)
                vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            elif shap_vals.ndim == 3:
                vals = shap_vals[0, :, 1]
            elif shap_vals.ndim == 2:
                vals = shap_vals[0]
            else:
                vals = shap_vals

            for name, val in zip(feature_names, vals):
                contributions[name] = float(np.round(val, 4))
        except Exception:
            contributions = _heuristic_attribution(raw_model, feature_names, X[0])
    else:
        contributions = _heuristic_attribution(raw_model, feature_names, X[0])

    # Partition into positive (risk increasing) and negative (risk mitigating)
    sorted_items = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    risk_drivers = [
        {"feature": k, "impact": v, "direction": "increases_risk"}
        for k, v in sorted_items if v > 0
    ][:top_k]

    mitigating_factors = [
        {"feature": k, "impact": v, "direction": "decreases_risk"}
        for k, v in sorted_items if v < 0
    ][:top_k]

    return {
        "top_risk_drivers": risk_drivers,
        "top_mitigating_factors": mitigating_factors,
        "all_contributions": contributions,
    }


def get_global_feature_importance(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    """Compute global feature importance ranking using mean absolute SHAP values."""
    raw_model = _extract_base_model(model)

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(raw_model)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            else:
                vals = shap_vals

            mean_abs = np.mean(np.abs(vals), axis=0)
            ranking = [
                {"feature": name, "importance": round(float(imp), 4)}
                for name, imp in zip(feature_names, mean_abs)
            ]
            return sorted(ranking, key=lambda x: x["importance"], reverse=True)
        except Exception:
            pass

    # Heuristic fallback using model feature_importances_ if available
    if hasattr(raw_model, "feature_importances_"):
        imps = raw_model.feature_importances_
        ranking = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in zip(feature_names, imps)
        ]
        return sorted(ranking, key=lambda x: x["importance"], reverse=True)

    # Uniform fallback
    return [{"feature": name, "importance": 1.0 / len(feature_names)} for name in feature_names]


def _heuristic_attribution(model: Any, feature_names: list[str], x_row: np.ndarray) -> dict[str, float]:
    """Fallback feature attribution when SHAP is not present."""
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        return {
            name: round(float(imp * (1.0 if x_row[i] > 0 else -0.5)), 4)
            for i, (name, imp) in enumerate(zip(feature_names, imps))
        }
    return {name: 0.0 for name in feature_names}
