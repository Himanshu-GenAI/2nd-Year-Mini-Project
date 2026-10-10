"""Unit tests for spatial gridding module."""

import numpy as np
import pytest

from src.config import GRID_SIZE_DEG
from src.data.grid import (
    coords_to_grid_id,
    generate_grid_mesh,
    get_grid_cell_bounds,
    get_neighboring_cells,
    grid_id_to_coords,
)


def test_coords_to_grid_id_single():
    lat, lon = 31.12, 77.24
    gid = coords_to_grid_id(lat, lon, grid_size=GRID_SIZE_DEG)
    assert isinstance(gid, str)
    assert gid.startswith("cell_")

    c_lat, c_lon = grid_id_to_coords(gid)
    # Centroid should be within half grid size of original coordinates
    assert abs(c_lat - lat) <= GRID_SIZE_DEG
    assert abs(c_lon - lon) <= GRID_SIZE_DEG


def test_coords_to_grid_id_vectorized():
    lats = np.array([30.0, 31.0, 32.0])
    lons = np.array([75.0, 76.0, 77.0])
    gids = coords_to_grid_id(lats, lons, grid_size=GRID_SIZE_DEG)
    assert len(gids) == 3
    for gid in gids:
        assert gid.startswith("cell_")


def test_get_grid_cell_bounds():
    gid = "cell_30.1250_75.1250"
    min_lat, max_lat, min_lon, max_lon = get_grid_cell_bounds(gid, grid_size=0.25)
    assert min_lat == pytest.approx(30.0, 0.001)
    assert max_lat == pytest.approx(30.25, 0.001)
    assert min_lon == pytest.approx(75.0, 0.001)
    assert max_lon == pytest.approx(75.25, 0.001)


def test_get_neighboring_cells():
    gid = "cell_30.1250_75.1250"
    neighbors = get_neighboring_cells(gid, grid_size=0.25, radius_steps=1)
    assert len(neighbors) == 8
    assert gid not in neighbors  # Self should not be in neighbors


def test_generate_grid_mesh():
    mesh = generate_grid_mesh(lat_min=30.0, lat_max=31.0, lon_min=75.0, lon_max=76.0, grid_size=0.5)
    assert len(mesh) == 4
    assert "grid_id" in mesh.columns
    assert "centroid_lat" in mesh.columns
    assert "centroid_lon" in mesh.columns
