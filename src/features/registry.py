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


# --- Day 3: Weather and KBDI Drought Features ---
_DAY3_FEATURES = [
    FeatureSpec(
        name="tmax_t0",
        group="weather",
        definition="Daily maximum 2m air temperature on day t.",
        window="t",
        unit="degC",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="tmax_t1",
        group="weather",
        definition="Daily maximum 2m air temperature on day t-1.",
        window="t-1",
        unit="degC",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="tmax_t3",
        group="weather",
        definition="Daily maximum 2m air temperature on day t-3.",
        window="t-3",
        unit="degC",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="tmax_t7",
        group="weather",
        definition="Daily maximum 2m air temperature on day t-7.",
        window="t-7",
        unit="degC",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="precip_t0",
        group="weather",
        definition="Daily precipitation sum on day t.",
        window="t",
        unit="mm",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="precip_t1",
        group="weather",
        definition="Daily precipitation sum on day t-1.",
        window="t-1",
        unit="mm",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="precip_t3",
        group="weather",
        definition="Daily precipitation sum on day t-3.",
        window="t-3",
        unit="mm",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="precip_t7",
        group="weather",
        definition="Daily precipitation sum on day t-7.",
        window="t-7",
        unit="mm",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="precip_sum_7d",
        group="weather",
        definition="Total precipitation over days t-6 to t (inclusive).",
        window="t-6..t",
        unit="mm",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="precip_sum_30d",
        group="weather",
        definition="Total precipitation over days t-29 to t (inclusive).",
        window="t-29..t",
        unit="mm",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="tmax_mean_7d",
        group="weather",
        definition="Mean daily maximum temperature over days t-6 to t (inclusive).",
        window="t-6..t",
        unit="degC",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="days_since_rain",
        group="weather",
        definition="Consecutive days up to t since last rainfall > 2.5 mm (0 on rain day, capped at 365).",
        window="2014-01-01..t",
        unit="days",
        dtype="float32",
        sources=("Open-Meteo Archive",),
        uses_future=False,
    ),
    FeatureSpec(
        name="kbdi",
        group="drought",
        definition="Keetch-Byram Drought Index (0-800) after day t with 2014 spin-up and 2015-2022 mean annual rainfall.",
        window="2014-01-01..t",
        unit="index_0_800",
        dtype="float32",
        sources=("Open-Meteo Archive", "USDA SE-38 KBDI"),
        uses_future=False,
    ),
    FeatureSpec(
        name="kbdi_change_7d",
        group="drought",
        definition="7-day change in KBDI: kbdi(t) - kbdi(t-7).",
        window="t-7..t",
        unit="delta_index",
        dtype="float32",
        sources=("Open-Meteo Archive", "USDA SE-38 KBDI"),
        uses_future=False,
    ),
]

for _spec in _DAY3_FEATURES:
    register_feature(_spec)

