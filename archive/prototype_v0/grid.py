"""Spatial Gridding System for Wildfire Discretization.

Discretizes continuous geographical coordinates (latitudes and longitudes)
into regular spatial mesh cells. Provides spatial neighbor lookups,
coordinate transformations, and bounding box aggregations.
"""


import numpy as np
import pandas as pd

from src.config import GRID_SIZE_DEG, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


def coords_to_grid_id(
    lat: float | np.ndarray | pd.Series,
    lon: float | np.ndarray | pd.Series,
    grid_size: float = GRID_SIZE_DEG,
) -> str | np.ndarray:
    """Snap geographic coordinates to regular grid cell identifier.

    The grid cell centroid is used as the cell identifier.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        grid_size: Grid resolution in decimal degrees.

    Returns:
        String grid identifier or array of strings: 'cell_<lat>_<lon>'.
    """
    half_size = grid_size / 2.0
    lat_centroid = np.floor(np.asarray(lat) / grid_size) * grid_size + half_size
    lon_centroid = np.floor(np.asarray(lon) / grid_size) * grid_size + half_size

    if np.ndim(lat) == 0 and np.ndim(lon) == 0:
        return f"cell_{lat_centroid:.4f}_{lon_centroid:.4f}"

    # Vectorized string formatting for pandas Series / numpy array
    return [
        f"cell_{c_lat:.4f}_{c_lon:.4f}"
        for c_lat, c_lon in zip(lat_centroid.flat, lon_centroid.flat)
    ]


def grid_id_to_coords(grid_id: str) -> tuple[float, float]:
    """Parse grid cell ID into centroid (latitude, longitude).

    Args:
        grid_id: Formatted string 'cell_<lat>_<lon>'.

    Returns:
        Tuple of (centroid_latitude, centroid_longitude).
    """
    parts = grid_id.split("_")
    if len(parts) != 3 or parts[0] != "cell":
        raise ValueError(f"Invalid grid identifier format: '{grid_id}'. Expected 'cell_<lat>_<lon>'.")
    return float(parts[1]), float(parts[2])


def get_grid_cell_bounds(
    grid_id: str,
    grid_size: float = GRID_SIZE_DEG,
) -> tuple[float, float, float, float]:
    """Calculate geographical bounding box for a given grid cell.

    Args:
        grid_id: Grid cell identifier.
        grid_size: Resolution in degrees.

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon).
    """
    c_lat, c_lon = grid_id_to_coords(grid_id)
    half = grid_size / 2.0
    return round(c_lat - half, 6), round(c_lat + half, 6), round(c_lon - half, 6), round(c_lon + half, 6)


def get_neighboring_cells(
    grid_id: str,
    grid_size: float = GRID_SIZE_DEG,
    radius_steps: int = 1,
) -> list[str]:
    """Find all adjacent Moore-neighborhood cells within radius steps.

    Args:
        grid_id: Center grid identifier.
        grid_size: Grid resolution in degrees.
        radius_steps: Number of grid steps away (default: 1 for 8 neighbors).

    Returns:
        List of neighboring grid IDs (excluding the center cell itself).
    """
    c_lat, c_lon = grid_id_to_coords(grid_id)
    neighbors = []

    for d_lat in range(-radius_steps, radius_steps + 1):
        for d_lon in range(-radius_steps, radius_steps + 1):
            if d_lat == 0 and d_lon == 0:
                continue
            n_lat = c_lat + d_lat * grid_size
            n_lon = c_lon + d_lon * grid_size
            neighbors.append(f"cell_{n_lat:.4f}_{n_lon:.4f}")

    return neighbors


def generate_grid_mesh(
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
    grid_size: float = GRID_SIZE_DEG,
) -> pd.DataFrame:
    """Generate exhaustive mesh of spatial grid cells over bounding box.

    Args:
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.
        grid_size: Step size in degrees.

    Returns:
        DataFrame containing grid_id, centroid_lat, centroid_lon, and bounding coordinates.
    """
    half = grid_size / 2.0
    lat_centroids = np.arange(lat_min + half, lat_max + half, grid_size)
    lon_centroids = np.arange(lon_min + half, lon_max + half, grid_size)

    grid_cells = []
    for c_lat in lat_centroids:
        for c_lon in lon_centroids:
            gid = f"cell_{c_lat:.4f}_{c_lon:.4f}"
            grid_cells.append(
                {
                    "grid_id": gid,
                    "centroid_lat": round(c_lat, 4),
                    "centroid_lon": round(c_lon, 4),
                    "min_lat": round(c_lat - half, 4),
                    "max_lat": round(c_lat + half, 4),
                    "min_lon": round(c_lon - half, 4),
                    "max_lon": round(c_lon + half, 4),
                }
            )

    return pd.DataFrame(grid_cells)
