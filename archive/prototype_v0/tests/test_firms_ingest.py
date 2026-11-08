"""Unit tests for FIRMS ingest and aggregation."""

import pandas as pd
import pytest

from src.data.firms_ingest import aggregate_firms_daily


def test_aggregate_firms_daily():
    mock_firms = pd.DataFrame(
        {
            "grid_id": ["cell_30.0_75.0", "cell_30.0_75.0", "cell_31.0_76.0"],
            "acq_date": pd.to_datetime(["2020-05-01", "2020-05-01", "2020-05-01"]),
            "latitude": [30.1, 30.2, 31.1],
            "longitude": [75.1, 75.2, 76.1],
            "frp": [15.2, 25.4, 8.1],
            "brightness": [320.0, 340.0, 310.0],
            "daynight": ["D", "D", "N"],
        }
    )

    agg = aggregate_firms_daily(mock_firms)

    # Should have 2 unique (grid_id, acq_date) rows
    assert len(agg) == 2

    # Check cell_30.0_75.0 aggregated metrics
    row_cell_30 = agg[agg["grid_id"] == "cell_30.0_75.0"].iloc[0]
    assert row_cell_30["fire_count"] == 2
    assert row_cell_30["max_frp"] == pytest.approx(25.4, 0.01)
    assert row_cell_30["sum_frp"] == pytest.approx(40.6, 0.01)
    assert row_cell_30["max_brightness"] == pytest.approx(340.0, 0.01)
    assert row_cell_30["day_fires"] == 2
    assert row_cell_30["night_fires"] == 0
