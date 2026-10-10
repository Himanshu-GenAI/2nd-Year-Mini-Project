"""Unit tests for wildfire baseline models, training, and probability calibration."""

import numpy as np
import pandas as pd

from src.models.baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    evaluate_model_predictions,
)
from src.models.calibrate import (
    compute_calibration_metrics,
)
from src.models.train import train_wildfire_model


def test_evaluate_model_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = evaluate_model_predictions(y_true, y_prob)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["roc_auc"] > 0.8


def test_persistence_baseline():
    X = pd.DataFrame({"fire_count_lag_1d": [0, 2, 0, 1]})
    model = PersistenceBaseline()
    probs = model.predict_proba(X)

    assert probs.shape == (4, 2)
    # Row with lag 2 should have high probability (0.85)
    assert probs[1, 1] == 0.85
    # Row with lag 0 should have low probability (0.05)
    assert probs[0, 1] == 0.05


def test_climatology_baseline():
    X_train = pd.DataFrame(
        {
            "grid_id": ["cell_A", "cell_A", "cell_B", "cell_B"],
            "month": [5, 5, 5, 5],
        }
    )
    y_train = pd.Series([1, 1, 0, 0])

    model = ClimatologyBaseline()
    model.fit(X_train, y_train)

    X_test = pd.DataFrame(
        {
            "grid_id": ["cell_A", "cell_B"],
            "month": [5, 5],
        }
    )
    probs = model.predict_proba(X_test)
    assert probs[0, 1] == 1.0  # Cell A always burned in May
    assert probs[1, 1] == 0.0  # Cell B never burned in May


def test_compute_calibration_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])
    cal_metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)

    assert "ece" in cal_metrics
    assert "brier_score" in cal_metrics
    assert cal_metrics["ece"] >= 0.0


def test_train_wildfire_model_end_to_end(tmp_path):
    # Construct small synthetic panel
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="W")
    n = len(dates)
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "date": dates,
            "grid_id": ["cell_10_20"] * n,
            "temp_max_c": rng.uniform(25, 40, n),
            "temp_min_c": rng.uniform(15, 25, n),
            "precip_24h_mm": rng.exponential(5, n),
            "relative_humidity_pct": rng.uniform(20, 80, n),
            "wind_speed_kmh": rng.uniform(10, 35, n),
            "kbdi": rng.uniform(100, 700, n),
            "kbdi_delta_7d": rng.normal(5, 10, n),
            "temp_max_rolling_mean_7d": rng.uniform(25, 40, n),
            "precip_rolling_sum_14d": rng.uniform(0, 50, n),
            "precip_rolling_sum_30d": rng.uniform(0, 100, n),
            "rh_rolling_min_7d": rng.uniform(15, 60, n),
            "wind_rolling_max_7d": rng.uniform(15, 45, n),
            "fire_count_lag_1d": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "fire_count_lag_3d": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "fire_count_lag_7d": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "fire_sum_30d": rng.integers(0, 5, n),
            "neighbor_fire_count_lag_1d": rng.choice([0, 1], n, p=[0.7, 0.3]),
            "day_of_year_sin": np.sin(2 * np.pi * dates.dayofyear / 365),
            "day_of_year_cos": np.cos(2 * np.pi * dates.dayofyear / 365),
            "month": dates.month,
            "fire_occurred_next_day": rng.choice([0, 1], n, p=[0.75, 0.25]),
        }
    )

    save_file = tmp_path / "model.joblib"
    model, summary = train_wildfire_model(
        df,
        calibrate=True,
        save_path=save_file,
    )

    assert save_file.exists()
    assert "test_metrics" in summary
    assert "pr_auc" in summary["test_metrics"]
