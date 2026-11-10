"""Feature registry documenting metadata, provenance, and leakage safety.

Maintains a central catalog of all model features across fire history, calendar,
static spatial attributes, weather, and drought indices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata specification for a single engineered feature."""

    name: str
    group: str  # 'fire', 'calendar', 'static', 'weather', 'drought'
    definition: str
    window: str
    unit: str
    dtype: str
    sources: tuple[str, ...]
    uses_future: bool = False


_FEATURE_REGISTRY: dict[str, FeatureSpec] = {}


def register_feature(spec: FeatureSpec) -> None:
    """Register a feature specification in the central registry."""
    if spec.uses_future:
        raise ValueError(f"Feature '{spec.name}' violates causality (uses_future=True).")
    _FEATURE_REGISTRY[spec.name] = spec


def registry_names() -> list[str]:
    """Return ordered list of all registered feature names."""
    return list(_FEATURE_REGISTRY.keys())


def get(name: str) -> FeatureSpec:
    """Retrieve FeatureSpec by feature name."""
    if name not in _FEATURE_REGISTRY:
        raise KeyError(f"Feature '{name}' not found in registry.")
    return _FEATURE_REGISTRY[name]


def all_specs() -> list[FeatureSpec]:
    """Return all registered FeatureSpec objects."""
    return list(_FEATURE_REGISTRY.values())


# --- Day 1: Fire History, Calendar & Static Features ---
_DAY1_FEATURES = [
    FeatureSpec(
        name="fire_count_7d",
        group="fire",
        definition="Total kept FIRMS detections in the cell over days t-6 to t (inclusive; partial window before 2015-01-07).",
        window="t-6..t",
        unit="count",
        dtype="float32",
        sources=("NASA FIRMS MODIS C6.1",),
        uses_future=False,
    ),
    FeatureSpec(
        name="fire_count_30d",
        group="fire",
        definition="Total kept FIRMS detections in the cell over days t-29 to t (inclusive; partial window before 2015-01-30).",
        window="t-29..t",
        unit="count",
        dtype="float32",
        sources=("NASA FIRMS MODIS C6.1",),
        uses_future=False,
    ),
    FeatureSpec(
        name="neighbour_fire_7d",
        group="fire",
        definition="Total kept FIRMS detections over days t-6 to t in the 8 adjacent Moore-neighborhood cells (kept cells only).",
        window="t-6..t",
        unit="count",
        dtype="float32",
        sources=("NASA FIRMS MODIS C6.1", "DataMeet Grid"),
        uses_future=False,
    ),
    FeatureSpec(
        name="month_sin",
        group="calendar",
        definition="Sine of cyclical month encoding: sin(2 * pi * month / 12).",
        window="t",
        unit="dimensionless",
        dtype="float32",
        sources=("Calendar date",),
        uses_future=False,
    ),
    FeatureSpec(
        name="month_cos",
        group="calendar",
        definition="Cosine of cyclical month encoding: cos(2 * pi * month / 12).",
        window="t",
        unit="dimensionless",
        dtype="float32",
        sources=("Calendar date",),
        uses_future=False,
    ),
    FeatureSpec(
        name="lat_c",
        group="static",
        definition="Grid cell centroid latitude in decimal degrees.",
        window="static",
        unit="degrees_north",
        dtype="float32",
        sources=("DataMeet Grid",),
        uses_future=False,
    ),
    FeatureSpec(
        name="lon_c",
        group="static",
        definition="Grid cell centroid longitude in decimal degrees.",
        window="static",
        unit="degrees_east",
        dtype="float32",
        sources=("DataMeet Grid",),
        uses_future=False,
    ),
]

for _spec in _DAY1_FEATURES:
    register_feature(_spec)
