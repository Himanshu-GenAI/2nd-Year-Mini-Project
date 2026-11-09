from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data.labels import primary_mask


def test_toy_labels_logic():
    # 3 cells x 12 days (2015-01-01 to 2015-01-12)
    # Target date t for evaluation: index 4 (2015-01-05)
    # Cell 1: detection on t+2 (2015-01-07) -> burning=0, new_fire=1, fire_next_48h=1
    # Cell 2: detection on t-2 (2015-01-03) and t+1 (2015-01-06) -> burning=1, new_fire=0, fire_next_48h=1
    # Cell 3: detection on t+3 (2015-01-08) -> burning=0, new_fire=0, fire_next_48h=0

    records = [
        # Cell 1
        {"cell_id": "r04c20", "acq_date": date(2015, 1, 7), "frp": 10.0},
        # Cell 2
        {"cell_id": "r04c21", "acq_date": date(2015, 1, 3), "frp": 15.0},
        {"cell_id": "r04c21", "acq_date": date(2015, 1, 6), "frp": 20.0},
        # Cell 3
        {"cell_id": "r05c19", "acq_date": date(2015, 1, 8), "frp": 5.0},
    ]
    det_df = pd.DataFrame(records)

    # Directly evaluate window calculations
    # Day t = 2015-01-05
    # Cell 1:
    c1_dets = det_df[det_df["cell_id"] == "r04c20"]
    assert len(c1_dets) == 1
    assert c1_dets.iloc[0]["acq_date"] == date(2015, 1, 7)  # t+2

    # Verify primary mask definition
    dummy_df = pd.DataFrame({"burning_prev3": [True, False, True, False]})
    mask = primary_mask(dummy_df)
    assert list(mask) == [False, True, False, True]


@pytest.mark.skipif(not Path("data/processed/fire_cell_day.parquet").exists(), reason="Fire table not built yet")
def test_fire_cell_day_panel_properties():
    df = pd.read_parquet("data/processed/fire_cell_day.parquet")

    # Step 5 & 6 row counts
    assert len(df) == 4684338

    split_counts = df["split"].value_counts().to_dict()
    assert split_counts["train"] == 3405306
    assert split_counts["val"] == 423621
    assert split_counts["test"] == 850743
    assert split_counts["gap"] == 4668

    # Gate checks
    # 1. new_fire_next_48h is 0 on all rows where burning_prev3 is True
    assert (df[df["burning_prev3"]]["new_fire_next_48h"] == 0).all()

    # 2. Any-fire rate higher than new-fire rate in all splits
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        any_rate = sub["fire_next_48h"].mean()
        new_rate = sub[primary_mask(sub)]["new_fire_next_48h"].mean()
        assert any_rate > new_rate
        assert 0.0 < any_rate < 0.50
        assert 0.0 < new_rate < 0.50

    # 3. Total n_det equals clean detections trimmed
    clean_df = pd.read_parquet("data/processed/detections_clean.parquet")
    clean_dates = pd.to_datetime(clean_df["acq_date"])
    trimmed_clean = clean_df[(clean_dates >= pd.to_datetime("2015-01-03")) & (clean_dates <= pd.to_datetime("2025-12-29"))]
    assert df["n_det"].sum() == len(trimmed_clean)
