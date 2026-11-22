"""Human-readable display labels and driver groupings for wildfire model features."""

from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    # Fires
    "fire_count_7d": "Fires in this cell, last 7 days",
    "fire_count_30d": "Fires in this cell, last 30 days",
    "neighbour_fire_7d": "Fires in nearby cells, last 7 days",
    # Geography & Calendar
    "month_sin": "Time of year",
    "month_cos": "Time of year",
    "lat_c": "Location",
    "lon_c": "Location",
    # Recent heat
    "tmax_t0": "Max temperature today",
    "tmax_t1": "Max temperature 1d ago",
    "tmax_t3": "Max temperature 3d ago",
    "tmax_t7": "Max temperature 7d ago",
    "tmax_mean_7d": "Mean max temp 7d",
    # Recent rainfall
    "precip_t0": "Precipitation today",
    "precip_t1": "Precipitation 1d ago",
    "precip_t3": "Precipitation 3d ago",
    "precip_t7": "Precipitation 7d ago",
    "precip_sum_7d": "Precipitation sum 7d",
    "precip_sum_30d": "Precipitation sum 30d",
    "days_since_rain": "Days since rain",
    # Drought index
    "kbdi": "Keetch-Byram Drought Index",
    "kbdi_change_7d": "KBDI change 7d",
}

GROUPS: dict[str, str] = {
    # Fires
    "fire_count_7d": FEATURE_LABELS["fire_count_7d"],
    "fire_count_30d": FEATURE_LABELS["fire_count_30d"],
    "neighbour_fire_7d": FEATURE_LABELS["neighbour_fire_7d"],
    # Baselines
    "lat_c": "Regional baseline",
    "lon_c": "Regional baseline",
    "month_sin": "Seasonal baseline",
    "month_cos": "Seasonal baseline",
    # Recent heat
    "tmax_t0": "Recent heat",
    "tmax_t1": "Recent heat",
    "tmax_t3": "Recent heat",
    "tmax_t7": "Recent heat",
    "tmax_mean_7d": "Recent heat",
    # Recent rainfall
    "precip_t0": "Recent rainfall",
    "precip_t1": "Recent rainfall",
    "precip_t3": "Recent rainfall",
    "precip_t7": "Recent rainfall",
    "precip_sum_7d": "Recent rainfall",
    "precip_sum_30d": "Recent rainfall",
    "days_since_rain": "Recent rainfall",
    # Drought index
    "kbdi": "Drought index",
    "kbdi_change_7d": "Drought index",
}

DRIVER_GROUPS_V1: list[str] = [
    "Fires in this cell, last 7 days",
    "Fires in this cell, last 30 days",
    "Fires in nearby cells, last 7 days",
    "Regional baseline",
    "Seasonal baseline",
    "Recent heat",
    "Recent rainfall",
    "Drought index",
]


def get_feature_label(feature: str) -> str:
    """Return human-readable label for a feature name."""
    if feature not in FEATURE_LABELS:
        raise KeyError(f"No human-readable label registered for feature {feature!r}")
    return FEATURE_LABELS[feature]


def get_driver_group(feature: str) -> str:
    """Return grouped driver label for a feature name."""
    if feature not in GROUPS:
        raise KeyError(f"No driver group registered for feature {feature!r}")
    return GROUPS[feature]
