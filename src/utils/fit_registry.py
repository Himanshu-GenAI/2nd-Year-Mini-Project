"""Fit registry to track and enforce training split boundary compliance.

Prevents temporal data leakage by recording the date boundaries of datasets
used to fit scalers, encoders, and baselines, verifying they do not exceed TRAIN_END.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.config import TRAIN_END

logger = logging.getLogger(__name__)


@dataclass
class FitRecord:
    name: str
    first_date: date
    last_date: date
    metadata: dict[str, Any] | None = None


_REGISTRY: list[FitRecord] = []


def register_fit(
    name: str,
    first_date: str | date,
    last_date: str | date,
    metadata: dict[str, Any] | None = None,
) -> FitRecord:
    """Register the date boundaries of a fitted model or transformation artifact."""
    d1 = date.fromisoformat(first_date) if isinstance(first_date, str) else first_date
    d2 = date.fromisoformat(last_date) if isinstance(last_date, str) else last_date

    record = FitRecord(name=name, first_date=d1, last_date=d2, metadata=metadata)
    _REGISTRY.append(record)
    return record


def get_registered_fits() -> list[FitRecord]:
    """Retrieve all currently registered fit records."""
    return list(_REGISTRY)


def clear_registry() -> None:
    """Clear in-memory fit registry."""
    _REGISTRY.clear()


def assert_fit_within_train(record: FitRecord) -> None:
    """Raise AssertionError if the record was fitted on dates later than TRAIN_END."""
    train_end_d = date.fromisoformat(TRAIN_END)
    if record.last_date > train_end_d:
        raise AssertionError(
            f"Leakage detected! Artifact '{record.name}' was fitted on data up to {record.last_date}, "
            f"which exceeds TRAIN_END ({train_end_d})."
        )
