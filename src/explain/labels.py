"""Human-readable display labels and driver groupings for wildfire model features."""

from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    "fire_count_7d": "Fires in this cell, last 7 days",
    "fire_count_30d": "Fires in this cell, last 30 days",
    "neighbour_fire_7d": "Fires in nearby cells, last 7 days",
    "month_sin": "Time of year",
    "month_cos": "Time of year",
    "lat_c": "Location",
    "lon_c": "Location",
}

GROUPS: dict[str, str] = {
    "fire_count_7d": FEATURE_LABELS["fire_count_7d"],
    "fire_count_30d": FEATURE_LABELS["fire_count_30d"],
    "neighbour_fire_7d": FEATURE_LABELS["neighbour_fire_7d"],
    "month_sin": "Seasonal baseline",
    "month_cos": "Seasonal baseline",
    "lat_c": "Regional baseline",
    "lon_c": "Regional baseline",
}


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
