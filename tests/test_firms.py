from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.config import CONF_MIN, FIRE_TYPE_KEEP, FIRMS_CSV_PATH
from src.data.firms import (
    apply_filters,
    assign_cells,
    build_clean_detections,
    drop_duplicates,
    load_clean_detections,
)
from src.data.grid import load_grid


def make_toy_firms_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": [30.25, 30.25, 30.25, 30.25, 1.0],  # 1.0 is outside grid
            "longitude": [78.75, 78.75, 78.75, 78.75, 70.0],
            "acq_date": [date(2020, 1, 1)] * 5,
            "acq_time": [1000, 1000, 1100, 1200, 1000],
            "confidence": [80, 80, 20, 80, 80],  # row 2 has conf 20 (< 30)
            "type": [0, 0, 0, 1, 0],  # row 3 has type 1 (!= 0)
            "satellite": ["Terra", "Terra", "Aqua", "Terra", "Terra"],
            "frp": [10.5, 10.5, 5.0, 12.0, 8.0],
        }
    )


def test_toy_deduplication():
    df = make_toy_firms_df()
    cleaned, n_removed = drop_duplicates(df)
    # Rows 0 and 1 are identical duplicates
    assert n_removed == 1
    assert len(cleaned) == 4


def test_toy_filters():
    df = make_toy_firms_df()
    # Row 2 (conf 20) and row 3 (type 1) must be dropped
    filtered, counts = apply_filters(df)
    assert counts["type_dropped"] == 1
    assert counts["conf_dropped"] == 1
    assert len(filtered) == 3
    assert (filtered["type"] == FIRE_TYPE_KEEP).all()
    assert (filtered["confidence"] >= CONF_MIN).all()


def test_toy_cell_assignment_and_outside_drop():
    df = pd.DataFrame(
        {
            "latitude": [30.25, 1.0],  # 30.25 is in grid; 1.0 is outside
            "longitude": [78.75, 70.0],
            "acq_date": [date(2020, 1, 1), date(2020, 1, 1)],
            "acq_time": [1000, 1000],
            "confidence": [80, 80],
            "type": [0, 0],
            "satellite": ["Terra", "Terra"],
            "frp": [10.0, 10.0],
        }
    )
    grid_df = load_grid()
    kept, n_dropped = assign_cells(df, grid_df)
    assert len(kept) == 1
    assert n_dropped == 1
    assert kept.iloc[0]["cell_id"] == "r48c23"


@pytest.mark.skipif(not FIRMS_CSV_PATH.exists(), reason="Raw FIRMS archive not available")
def test_full_pipeline_audit_counts():
    df_clean = load_clean_detections()
    assert len(df_clean) == 811280

    # Read audit trail
    audit_file = Path("docs/audit/firms_load_counts.csv")
    assert audit_file.exists()
    audit_df = pd.read_csv(audit_file).set_index("step")

    assert audit_df.loc["raw", "count"] == 923584
    assert audit_df.loc["after_filters", "count"] == 834958
    assert audit_df.loc["after_cell_assignment", "count"] == 811280

    # Verification of share inside kept cells: 811280 / 834958 = 97.164%
    share = audit_df.loc["after_cell_assignment", "count"] / audit_df.loc["after_filters", "count"]
    # docs/audit/grid_coverage.csv records 0.9716
    assert abs(share - 0.9716) < 0.0005


def test_build_clean_detections_fast_path(tmp_path: Path):
    # Test on a small slice
    toy_csv = tmp_path / "toy_firms.csv"
    toy_df = make_toy_firms_df()
    toy_df.to_csv(toy_csv, index=False)

    out_file = tmp_path / "toy_clean.parquet"
    res = build_clean_detections(csv_path=toy_csv, output_path=out_file, write_docs_audit=False)
    assert len(res) == 1
    assert out_file.exists()
    loaded = load_clean_detections(out_file)
    assert len(loaded) == 1
    assert loaded.iloc[0]["cell_id"] == "r48c23"
