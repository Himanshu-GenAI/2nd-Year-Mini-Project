"""Baseline Wildfire Predictors and Evaluation Benchmarks.

Implements standard benchmark heuristics against which complex ML architectures are compared:
1. Persistence Baseline: Predicts fire today if cell had fire yesterday.
2. Climatology Baseline: Predicts empirical historical monthly frequency per cell.
3. Prior Baseline: Predicts global training prior probability.
4. Logistic Regression Baseline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate comprehensive evaluation metrics for wildfire risk classification.

    Given the high class imbalance in wildfire events (often < 5% positive),
    PR-AUC (Average Precision) and Brier Score are prioritized over raw accuracy.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        threshold: Decision cutoff for hard binary classifications.

    Returns:
        Dictionary of formatted evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics with zero division protection
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float(np.mean(y_true))

    brier = float(brier_score_loss(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
    }


class PersistenceBaseline:
    """Predicts next-day fire presence if fire occurred in the cell on the previous day."""

    def __init__(self, lag_col: str = "fire_count_lag_1d"):
        self.lag_col = lag_col

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PersistenceBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        fires = (X[self.lag_col] > 0).astype(float).values
        # Probabilities: 0.85 if burned yesterday, 0.05 otherwise
        probs = np.where(fires, 0.85, 0.05)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.lag_col] > 0).astype(int).values


class ClimatologyBaseline:
    """Predicts empirical historical frequency of fires for each (grid_id, month) tuple."""

    def __init__(self, grid_col: str = "grid_id", month_col: str = "month"):
        self.grid_col = grid_col
        self.month_col = month_col
        self.climatology_map: dict[tuple[str, int], float] = {}
        self.global_prior: float = 0.05

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClimatologyBaseline":
        df = X[[self.grid_col, self.month_col]].copy()
        df["target"] = y.values
        grouped = df.groupby([self.grid_col, self.month_col])["target"].mean()
        self.climatology_map = grouped.to_dict()
        self.global_prior = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = []
        for gid, m in zip(X[self.grid_col], X[self.month_col]):
            p = self.climatology_map.get((gid, int(m)), self.global_prior)
            probs.append(p)
        probs = np.array(probs)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class LogisticRegressionBaseline:
    """Standard L2-regularized logistic regression benchmark on normalized features."""

    def __init__(self, c_val: float = 1.0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=c_val, class_weight="balanced", max_iter=500, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionBaseline":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
