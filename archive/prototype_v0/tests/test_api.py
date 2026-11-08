"""Unit and integration tests for FastAPI wildfire serving endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["model_loaded"] is True


def test_info_endpoint(client):
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "grid_resolution_deg" in data
    assert "features" in data
    assert len(data["features"]) > 0


def test_predict_endpoint(client):
    payload = {
        "latitude": 31.5,
        "longitude": 77.2,
        "features": {
            "temp_max_c": 38.0,
            "temp_min_c": 26.0,
            "precip_24h_mm": 0.0,
            "relative_humidity_pct": 25.0,
            "wind_speed_kmh": 28.0,
            "kbdi": 550.0,
            "kbdi_delta_7d": 20.0,
            "temp_max_rolling_mean_7d": 37.5,
            "precip_rolling_sum_14d": 0.0,
            "precip_rolling_sum_30d": 5.0,
            "rh_rolling_min_7d": 20.0,
            "wind_rolling_max_7d": 32.0,
            "fire_count_lag_1d": 1,
            "fire_count_lag_3d": 0,
            "fire_count_lag_7d": 1,
            "fire_sum_30d": 3,
            "neighbor_fire_count_lag_1d": 2,
            "day_of_year_sin": 0.8,
            "day_of_year_cos": -0.6,
            "month": 5,
        },
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "grid_id" in data
    assert data["grid_id"].startswith("cell_")
    assert 0.0 <= data["fire_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Moderate", "High", "Extreme"]


def test_explain_endpoint(client):
    payload = {
        "latitude": 31.5,
        "longitude": 77.2,
        "features": {
            "temp_max_c": 38.0,
            "temp_min_c": 26.0,
            "precip_24h_mm": 0.0,
            "relative_humidity_pct": 25.0,
            "wind_speed_kmh": 28.0,
            "kbdi": 550.0,
            "kbdi_delta_7d": 20.0,
            "temp_max_rolling_mean_7d": 37.5,
            "precip_rolling_sum_14d": 0.0,
            "precip_rolling_sum_30d": 5.0,
            "rh_rolling_min_7d": 20.0,
            "wind_rolling_max_7d": 32.0,
            "fire_count_lag_1d": 1,
            "fire_count_lag_3d": 0,
            "fire_count_lag_7d": 1,
            "fire_sum_30d": 3,
            "neighbor_fire_count_lag_1d": 2,
            "day_of_year_sin": 0.8,
            "day_of_year_cos": -0.6,
            "month": 5,
        },
    }

    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_risk_drivers" in data
    assert "top_mitigating_factors" in data
