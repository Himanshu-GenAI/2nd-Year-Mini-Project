"""Baseline Wildfire Forecasting Models.

Implements historical empirical climatology baseline with Laplace smoothing,
hierarchical fallbacks (cell-month -> cell -> global), and fit registry tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.data.labels import primary_mask
from src.utils.fit_registry import register_fit


@dataclass
class ClimatologyModel:
    """Stores fitted historical frequency tables and fallback priors."""

    label: str
    cell_month_rates: dict[tuple[str, int], float]
    cell_rates: dict[str, float]
    global_rate: float
    first_date: date
    last_date: date


def fit_climatology(
    train_df: pd.DataFrame,
    label: str,
    alpha: float = 1.0,
) -> ClimatologyModel:
    """Fit empirical historical climatology baseline with Laplace smoothing.

    For each (cell_id, month), computes:
        rate = (positives + alpha) / (rows + 2 * alpha)

    Args:
        train_df: Training split DataFrame.
        label: Target label column name ('fire_next_48h' or 'new_fire_next_48h').
        alpha: Laplace smoothing parameter (default 1.0).

    Returns:
        Fitted ClimatologyModel.
    """
    df = train_df.copy()
    if label == "new_fire_next_48h":
        df = df[primary_mask(df)].copy()

    first_date = pd.to_datetime(df["date"]).min().date()
    last_date = pd.to_datetime(df["date"]).max().date()

    # Register fit in fit registry
    register_fit(
        name=f"climatology_{label}",
        first_date=first_date,
        last_date=last_date,
        metadata={"alpha": alpha, "label": label},
    )

    df["month"] = pd.to_datetime(df["date"]).dt.month

    # 1. Cell-month rates
    cm_group = df.groupby(["cell_id", "month"], observed=True)[label]
    cm_pos = cm_group.sum()
    cm_rows = cm_group.count()
    cm_rates = ((cm_pos + alpha) / (cm_rows + 2.0 * alpha)).to_dict()

    # 2. Per-cell overall rates
    c_group = df.groupby("cell_id", observed=True)[label]
    c_pos = c_group.sum()
    c_rows = c_group.count()
    c_rates = ((c_pos + alpha) / (c_rows + 2.0 * alpha)).to_dict()

    # 3. Global overall rate
    g_pos = float(df[label].sum())
    g_rows = float(len(df))
    g_rate = (g_pos + alpha) / (g_rows + 2.0 * alpha)

    return ClimatologyModel(
        label=label,
        cell_month_rates=cm_rates,
        cell_rates=c_rates,
        global_rate=g_rate,
        first_date=first_date,
        last_date=last_date,
    )


def predict_climatology(model: ClimatologyModel, df: pd.DataFrame) -> np.ndarray:
    """Predict risk scores for each (cell_id, date) row using fitted climatology."""
    months = pd.to_datetime(df["date"]).dt.month.values
    cells = df["cell_id"].astype(str).values

    scores = np.empty(len(df), dtype=np.float32)

    for i, (c, m) in enumerate(zip(cells, months)):
        # Check cell-month
        key = (c, int(m))
        if key in model.cell_month_rates:
            scores[i] = model.cell_month_rates[key]
        elif c in model.cell_rates:
            scores[i] = model.cell_rates[c]
        else:
            scores[i] = model.global_rate

    return scores
