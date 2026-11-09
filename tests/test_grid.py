import numpy as np
import pytest

from src.config import GRID_DEG, GRID_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
from src.data.grid import cell_row_col, load_grid, make_cell_id


def test_spot_check_reference_cell():
    row, col = cell_row_col(30.25, 78.75)
    assert row == 48
    assert col == 23
    assert make_cell_id(row, col) == "r48c23"


def test_lower_boundary_inclusion():
    # Lower edge of row 48, col 23 is lat=30.0, lon=78.5
    row, col = cell_row_col(30.0, 78.5)
    assert row == 48
    assert col == 23
    assert make_cell_id(row, col) == "r48c23"


def test_upper_boundary_is_outside():
    # Exactly on upper boundary: lat=38.0 or lon=98.0
    row, col = cell_row_col(LAT_MAX, 75.0)
    assert row == -1
    assert col == -1
    assert make_cell_id(row, col) == ""

    row2, col2 = cell_row_col(20.0, LON_MAX)
    assert row2 == -1
    assert col2 == -1


def test_out_of_bounds_points():
    lats = np.array([5.9, 38.1, 20.0, 20.0])
    lons = np.array([75.0, 75.0, 66.9, 98.1])
    rows, cols = cell_row_col(lats, lons)
    assert (rows == -1).all()
    assert (cols == -1).all()


def test_candidate_corners_and_vectorization():
    # Corner 0,0: lat=6.0, lon=67.0
    r0, c0 = cell_row_col(LAT_MIN, LON_MIN)
    assert r0 == 0 and c0 == 0
    assert make_cell_id(r0, c0) == "r00c00"

    # Corner 63,61: top-right candidate cell starts at lat=37.5, lon=97.5
    r_top, c_top = cell_row_col(LAT_MAX - GRID_DEG / 2, LON_MAX - GRID_DEG / 2)
    assert r_top == 63 and c_top == 61
    assert make_cell_id(r_top, c_top) == "r63c61"


def test_make_cell_id_array():
    rows = np.array([0, 48, -1])
    cols = np.array([0, 23, -1])
    ids = make_cell_id(rows, cols)
    assert list(ids) == ["r00c00", "r48c23", ""]


@pytest.mark.skipif(not GRID_PATH.exists(), reason="Grid file not present in git")
def test_load_grid_validation():
    gdf = load_grid()
    assert len(gdf) == 1167
    assert gdf["cell_id"].nunique() == 1167
    assert (gdf["cell_id"] == make_cell_id(gdf["row"].values, gdf["col"].values)).all()
