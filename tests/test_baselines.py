from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.models.baselines import fit_climatology, predict_climatology
from src.models.dataset import load_split
from src.utils.fit_registry import assert_fit_within_train, get_registered_fits


def test_toy_climatology_smoothing_and_fallbacks():
    # 2 cells, months 1 and 2
    # cell 1, month 1: 3 rows, 1 positive. With alpha=1: (1 + 1) / (3 + 2) = 2/5 = 0.40.
    # cell 1, month 2: 1 row, 0 positive. With alpha=1: (0 + 1) / (1 + 2) = 1/3 = 0.3333.
    # cell 2, month 1: 2 rows, 2 positives. With alpha=1: (2 + 1) / (2 + 2) = 3/4 = 0.75.
    train_df = pd.DataFrame(
        {
            "cell_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
            "date": [
                date(2020, 1, 1),
                date(2020, 1, 2),
                date(2020, 1, 3),
                date(2020, 2, 1),
                date(2020, 1, 1),
                date(2020, 1, 2),
            ],
            "burning_prev3": [False] * 6,
            "fire_next_48h": [1, 0, 0, 0, 1, 1],
        }
    )

    model = fit_climatology(train_df, "fire_next_48h", alpha=1.0)

    assert abs(model.cell_month_rates[("c1", 1)] - 0.40) < 1e-5
    assert abs(model.cell_month_rates[("c1", 2)] - 1.0 / 3.0) < 1e-5
    assert abs(model.cell_month_rates[("c2", 1)] - 0.75) < 1e-5

    # Test prediction with known values and fallbacks
    eval_df = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3"],  # c2 month 2 missing -> cell fallback; c3 unknown -> global fallback
            "date": [date(2021, 1, 15), date(2021, 2, 10), date(2021, 5, 1)],
        }
    )
    preds = predict_climatology(model, eval_df)
    assert abs(preds[0] - 0.40) < 1e-5
    # c2 cell fallback: 2 rows, 2 pos -> (2+1)/(2+2) = 0.75
    assert abs(preds[1] - 0.75) < 1e-5
    # c3 global fallback: 6 rows, 3 pos -> (3+1)/(6+2) = 4/8 = 0.50
    assert abs(preds[2] - 0.50) < 1e-5


def test_fit_record_registered_and_within_train():
    train_df = pd.DataFrame(
        {
            "cell_id": ["c1", "c1"],
            "date": [date(2015, 1, 3), date(2022, 12, 29)],
            "burning_prev3": [False, False],
            "fire_next_48h": [0, 1],
        }
    )
    model = fit_climatology(train_df, "fire_next_48h", alpha=1.0)
    assert model.last_date <= date(2022, 12, 29)

    fits = [r for r in get_registered_fits() if r.name == "climatology_fire_next_48h"]
    assert len(fits) > 0
    assert_fit_within_train(fits[-1])


@pytest.mark.skipif(not Path("data/processed/fire_cell_day.parquet").exists(), reason="Fire table not built yet")
def test_load_split_guardrails():
    # Gap split always raises ValueError
    with pytest.raises(ValueError, match="prohibited"):
        load_split("gap", "fire_next_48h")

    # Test split raises without ALLOW_TEST_SPLIT=1
    with pytest.raises(PermissionError, match="locked"):
        load_split("test", "fire_next_48h")

    # Primary set has no burning_prev3 == True
    val_primary = load_split("val", "new_fire_next_48h", primary=True)
    assert not val_primary["burning_prev3"].any()
