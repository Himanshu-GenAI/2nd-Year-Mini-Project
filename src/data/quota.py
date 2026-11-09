"""Quota ledger for Open-Meteo API requests.

Tracks request weights across rolling 1-minute, 1-hour, and 24-hour windows.
Provides atomic persistence, rate limiting computations, and quota safety caps.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import (
    LEDGER_PATH,
    QUOTA_DAY_CAP,
    QUOTA_HOUR_CAP,
    QUOTA_MINUTE_CAP,
)

logger = logging.getLogger(__name__)

WINDOW_1M_S = 60.0
WINDOW_1H_S = 3600.0
WINDOW_24H_S = 86400.0


class QuotaLedger:
    """Manages rolling rate-limit quota state for Open-Meteo API requests."""

    def __init__(
        self,
        ledger_path: Path | str = LEDGER_PATH,
        clock: Callable[[], float] | None = None,
        minute_cap: float = QUOTA_MINUTE_CAP,
        hour_cap: float = QUOTA_HOUR_CAP,
        day_cap: float = QUOTA_DAY_CAP,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        self.clock = clock or time.time
        self.minute_cap = minute_cap
        self.hour_cap = hour_cap
        self.day_cap = day_cap
        self.entries: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        """Load ledger entries from disk and prune entries older than 24 hours."""
        if not self.ledger_path.exists():
            self.entries = []
            return

        try:
            with open(self.ledger_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.entries = data
            else:
                logger.warning("Corrupted ledger format (not a list). Starting with empty ledger.")
                self.entries = []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse ledger file (%s). Starting with empty ledger.", e)
            self.entries = []

        now = self.clock()
        self.prune(now)

    def prune(self, now: float) -> None:
        """Discard entries older than 24 hours."""
        cutoff = now - WINDOW_24H_S
        self.entries = [e for e in self.entries if e.get("timestamp", 0.0) >= cutoff]

    def save(self) -> None:
        """Atomically persist ledger entries to disk."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        dir_name = self.ledger_path.parent
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            temp_name = tf.name
            json.dump(self.entries, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())

        os.replace(temp_name, self.ledger_path)

    def used(self, window_seconds: float, now: float | None = None) -> float:
        """Calculate total weight consumed in the last `window_seconds`."""
        current_time = self.clock() if now is None else now
        cutoff = current_time - window_seconds
        return sum(e["weight"] for e in self.entries if e.get("timestamp", 0.0) >= cutoff)

    def seconds_until_allowed(self, weight: float, now: float | None = None) -> float:
        """Return minimum seconds to wait until `weight` can be added without exceeding caps."""
        current_time = self.clock() if now is None else now
        self.prune(current_time)

        windows = [
            (WINDOW_1M_S, self.minute_cap),
            (WINDOW_1H_S, self.hour_cap),
            (WINDOW_24H_S, self.day_cap),
        ]

        max_wait = 0.0

        for window_s, cap in windows:
            cutoff = current_time - window_s
            in_window = [e for e in self.entries if e.get("timestamp", 0.0) >= cutoff]
            in_window.sort(key=lambda x: x["timestamp"])

            current_used = sum(e["weight"] for e in in_window)
            if current_used + weight <= cap:
                continue

            excess = (current_used + weight) - cap
            accumulated = 0.0
            window_wait = 0.0

            for entry in in_window:
                accumulated += entry["weight"]
                if accumulated >= excess:
                    window_wait = max(0.0, (entry["timestamp"] + window_s) - current_time)
                    break
            else:
                window_wait = window_s

            if window_wait > max_wait:
                max_wait = window_wait

        return max_wait

    def record(self, weight: float, cell_id: str, status: int, now: float | None = None) -> None:
        """Record a completed request entry and atomically save."""
        current_time = self.clock() if now is None else now
        self.prune(current_time)
        entry = {
            "timestamp": current_time,
            "weight": float(weight),
            "cell_id": str(cell_id),
            "status": int(status),
        }
        self.entries.append(entry)
        self.save()
