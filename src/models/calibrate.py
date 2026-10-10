"""Probability Calibration and Reliability Assessment for Wildfire Risk.

Post-processes classification outputs using Isotonic Regression and Platt Scaling (Sigmoid)
so that predicted risk scores correspond directly to empirical fire probabilities.
Calculates Expected Calibration Error (ECE) and reliability curve statistics.
"""

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and reliability curve points.

    Args:
        y_true: True binary targets (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of probability bins (default: 10).

    Returns:
        Dictionary containing:
        - ece: Expected Calibration Error (weighted average difference).
        - mce: Maximum Calibration Error.
        - brier_score: Overall probability mean squared error.
        - prob_true: Empirical fraction of positives per bin.
        - prob_pred: Mean predicted probability per bin.
        - bin_counts: Number of samples per bin.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_true)

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Calculate binned metrics for exact ECE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    bin_weights = []
    bin_errors = []

    for b in range(n_bins):
        mask = bin_assignments == b
        count = np.sum(mask)
        if count > 0:
            avg_pred = np.mean(y_prob[mask])
            avg_true = np.mean(y_true[mask])
            error = np.abs(avg_pred - avg_true)
            bin_weights.append(count / n)
            bin_errors.append(error)

    ece = float(np.sum(np.array(bin_weights) * np.array(bin_errors))) if bin_weights else 0.0
    mce = float(np.max(bin_errors)) if bin_errors else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


class CalibratedWildfireModel:
    """Wrapper that fits or evaluates probability calibration on top of an estimator.

    Supports Isotonic Regression (non-parametric, flexible) and Sigmoid (Platt scaling).
    """

    def __init__(
        self,
        base_estimator: Any,
        method: str = "isotonic",
    ):
        self.base_estimator = base_estimator
        self.method = method
        self.calibrated_model: CalibratedClassifierCV | None = None

    def fit_calibration(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "CalibratedWildfireModel":
        """Calibrate using a distinct temporal validation dataset.

        Uses cv='prefit' to preserve base estimator's learned decision boundaries.
        """
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_estimator,
            method=self.method,
            cv="prefit",
        )
        self.calibrated_model.fit(X_val, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        return self.base_estimator.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
