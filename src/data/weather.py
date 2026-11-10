"""Resumable Open-Meteo weather fetcher with quota rate limiting.

Downloads historical daily maximum temperature and precipitation sums
for India spatial grid cells (2014-2025) while strictly honoring API rate caps.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    CELL_WEIGHT,
    GRID_PATH,
    MIN_INTERVAL_S,
    PROJECT_ROOT,
    SHUFFLE_SEED,
    WEATHER_API_URL,
    WEATHER_DIR,
    WEATHER_END,
    WEATHER_N_DAYS,
    WEATHER_START,
    WEATHER_VARS,
)
from src.data.quota import QuotaLedger

logger = logging.getLogger(__name__)

LOCK_FILE = WEATHER_DIR.parent / "fetch.lock"
FAILED_LOG = WEATHER_DIR.parent / "weather_failed.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "weather_fetch.log"


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is currently active."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # PROCESS_QUERY_LIMITED_INFORMATION (0x1000)
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


class WeatherFetcher:
    """Orchestrates compliant downloading, validation, and storage of cell weather."""

    def __init__(
        self,
        grid_path: Path = GRID_PATH,
        weather_dir: Path = WEATHER_DIR,
        ledger: QuotaLedger | None = None,
        session: requests.Session | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        failed_log: Path = FAILED_LOG,
        log_file: Path = LOG_FILE,
    ) -> None:
        self.grid_path = Path(grid_path)
        self.weather_dir = Path(weather_dir)
        self.weather_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger or QuotaLedger()
        self.session = session or requests.Session()
        self.clock = clock or time.time
        self.sleep_fn = sleep_fn or time.sleep
        self.failed_log = Path(failed_log)
        self.log_file = Path(log_file)
        self.last_request_start: float = 0.0

    def get_cell_list(self) -> list[dict[str, Any]]:
        """Load grid cells, sort by cell_id, and deterministically shuffle."""
        if not self.grid_path.exists():
            raise FileNotFoundError(f"Grid file not found at {self.grid_path}")

        df = pd.read_parquet(self.grid_path)
        cells = df[["cell_id", "lat_c", "lon_c"]].to_dict(orient="records")
        cells.sort(key=lambda c: str(c["cell_id"]))

        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(cells)
        return cells

    def is_cell_cached(self, cell_id: str) -> bool:
        """Verify if a cell Parquet exists and contains exactly WEATHER_N_DAYS rows."""
        file_path = self.weather_dir / f"{cell_id}.parquet"
        if not file_path.exists():
            return False

        try:
            df = pd.read_parquet(file_path)
            if len(df) == WEATHER_N_DAYS:
                return True
            logger.warning("Cell %s parquet has %d rows (expected %d). Re-fetching.", cell_id, len(df), WEATHER_N_DAYS)
            file_path.unlink(missing_ok=True)
            return False
        except Exception as e:
            logger.warning("Cell %s parquet corrupted (%s). Re-fetching.", cell_id, e)
            file_path.unlink(missing_ok=True)
            return False

    def validate_payload(self, data: dict[str, Any]) -> pd.DataFrame:
        """Validate Open-Meteo API payload structure, continuity, and value ranges."""
        if "daily" not in data:
            raise ValueError("Missing 'daily' section in response")

        daily = data["daily"]
        for key in ("time", "temperature_2m_max", "precipitation_sum"):
            if key not in daily:
                raise ValueError(f"Missing '{key}' in daily payload")

        times = daily["time"]
        tmax = daily["temperature_2m_max"]
        precip = daily["precipitation_sum"]

        if len(times) != WEATHER_N_DAYS or len(tmax) != WEATHER_N_DAYS or len(precip) != WEATHER_N_DAYS:
            raise ValueError(f"Payload row count mismatch. Expected {WEATHER_N_DAYS}, got {len(times)}")

        if any(v is None for v in tmax) or any(v is None for v in precip):
            raise ValueError("Payload contains null values in required variables")

        # Verify continuous dates
        start_d = date.fromisoformat(WEATHER_START)
        end_d = date.fromisoformat(WEATHER_END)
        expected_dates = [start_d + timedelta(days=i) for i in range(WEATHER_N_DAYS)]
        actual_dates = [date.fromisoformat(t) for t in times]

        if actual_dates[0] != start_d or actual_dates[-1] != end_d or actual_dates != expected_dates:
            raise ValueError("Dates are not continuous from WEATHER_START to WEATHER_END")

        # Range checks
        for val in tmax:
            if not (-20.0 <= float(val) <= 60.0):
                raise ValueError(f"Temperature {val}°C out of valid range [-20, 60]")

        for val in precip:
            if float(val) < 0.0:
                raise ValueError(f"Precipitation {val} mm is negative")

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(actual_dates).date,
                "temperature_2m_max": pd.Series(tmax, dtype="float32"),
                "precipitation_sum": pd.Series(precip, dtype="float32"),
            }
        )
        return df

    def save_cell_parquet(self, cell_id: str, df: pd.DataFrame) -> None:
        """Atomically save validated weather DataFrame to Parquet."""
        df = df.copy()
        df["cell_id"] = cell_id
        df["fetched_at"] = datetime.now(timezone.utc).isoformat()

        final_path = self.weather_dir / f"{cell_id}.parquet"
        tmp_path = self.weather_dir / f"{cell_id}.tmp.parquet"

        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, final_path)

    def log_failure(self, cell_id: str, reason: str) -> None:
        """Record cell fetch failure to CSV ledger."""
        self.failed_log.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.failed_log.exists()
        with open(self.failed_log, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["cell_id", "reason", "timestamp_utc"])
            writer.writerow([cell_id, reason, datetime.now(timezone.utc).isoformat()])

    def pace(self) -> None:
        """Enforce quota ledger waits and minimum inter-request interval."""
        now = self.clock()
        quota_wait = self.ledger.seconds_until_allowed(CELL_WEIGHT, now=now)

        interval_wait = 0.0
        if self.last_request_start > 0.0:
            elapsed = now - self.last_request_start
            if elapsed < MIN_INTERVAL_S:
                interval_wait = MIN_INTERVAL_S - elapsed

        total_wait = max(quota_wait, interval_wait)
        if total_wait > 0.0:
            remaining = total_wait
            while remaining > 0.0:
                chunk = min(remaining, 30.0)
                if chunk > 10.0:
                    logger.info("Pacing: sleeping for %.1fs (%.1fs remaining)...", chunk, remaining)
                self.sleep_fn(chunk)
                remaining -= chunk

    def fetch_cell(self, cell: dict[str, Any]) -> tuple[bool, int, str]:
        """Fetch weather data for a single cell with retries and rate limit protection."""
        cell_id = str(cell["cell_id"])
        lat = round(float(cell["lat_c"]), 2)
        lon = round(float(cell["lon_c"]), 2)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": WEATHER_START,
            "end_date": WEATHER_END,
            "daily": ",".join(WEATHER_VARS),
            "timezone": "GMT",
        }
        headers = {
            "User-Agent": "WildfireEarlyAlert/1.0 (PS-113; Student Research Project)",
        }

        retries_429 = 0
        retries_other = 0

        while True:
            self.pace()
            self.last_request_start = self.clock()
            start_t = self.clock()

            try:
                resp = self.session.get(WEATHER_API_URL, params=params, headers=headers, timeout=60.0)
                duration = self.clock() - start_t
                status_code = resp.status_code

                # Handle 429 Rate Limit
                if status_code == 429:
                    reason_text = resp.text
                    logger.warning("HTTP 429 Rate Limited on cell %s: %s", cell_id, reason_text)
                    self.ledger.record(0.0, cell_id, 429, now=self.clock())

                    if "daily" in reason_text.lower():
                        return False, 429, f"Daily limit exceeded: {reason_text}"

                    retries_429 += 1
                    if retries_429 > 3:
                        return False, 429, f"Exceeded max 429 retries: {reason_text}"

                    wait_s = 60.0
                    logger.info("Waiting %.1fs before retrying 429 on cell %s...", wait_s, cell_id)
                    self.sleep_fn(wait_s)
                    continue

                # Handle 5xx / Network HTTP errors
                if status_code >= 500 or status_code != 200:
                    retries_other += 1
                    if retries_other > 3:
                        msg = f"HTTP {status_code}: {resp.text[:100]}"
                        self.log_failure(cell_id, msg)
                        return False, status_code, msg

                    delay = [10.0, 30.0, 90.0][min(retries_other - 1, 2)]
                    logger.warning("HTTP %d on cell %s. Retrying in %.1fs...", status_code, cell_id, delay)
                    self.sleep_fn(delay)
                    continue

                # 200 OK: Parse, validate, and write
                payload = resp.json()
                df = self.validate_payload(payload)
                self.save_cell_parquet(cell_id, df)
                self.ledger.record(CELL_WEIGHT, cell_id, 200, now=self.clock())

                # Log success to log file
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as lf:
                    lf.write(
                        f"{datetime.now(timezone.utc).isoformat()} | cell={cell_id} | status=200 | "
                        f"duration={duration:.2f}s | weight={CELL_WEIGHT:.2f} | "
                        f"1m_used={self.ledger.used(60.0):.1f} | 1h_used={self.ledger.used(3600.0):.1f} | "
                        f"24h_used={self.ledger.used(86400.0):.1f}\n"
                    )

                return True, 200, "OK"

            except Exception as e:
                retries_other += 1
                if retries_other > 3:
                    msg = f"Exception: {e}"
                    self.log_failure(cell_id, msg)
                    return False, 500, msg

                delay = [10.0, 30.0, 90.0][min(retries_other - 1, 2)]
                logger.warning("Network exception on cell %s (%s). Retrying in %.1fs...", cell_id, e, delay)
                self.sleep_fn(delay)


def acquire_lock(lock_file: Path = LOCK_FILE) -> bool:
    """Acquire PID lock to prevent concurrent fetch processes."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    if lock_file.exists():
        try:
            pid = int(lock_file.read_text(encoding="utf-8").strip())
            if is_pid_alive(pid):
                return False
        except (ValueError, OSError):
            pass

    lock_file.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock(lock_file: Path = LOCK_FILE) -> None:
    """Release the PID lock."""
    lock_file.unlink(missing_ok=True)


def print_status(fetcher: WeatherFetcher) -> None:
    """Print cached cells count, failures, ledger usage, and ETA."""
    cells = fetcher.get_cell_list()
    total_cells = len(cells)
    cached_count = sum(1 for c in cells if fetcher.is_cell_cached(str(c["cell_id"])))

    failed_count = 0
    if FAILED_LOG.exists():
        try:
            df_fail = pd.read_csv(FAILED_LOG)
            failed_count = len(df_fail)
        except Exception:
            pass

    remaining = total_cells - cached_count
    eta_days = remaining / 127.0 if remaining > 0 else 0.0

    print("=" * 60)
    print("OPEN-METEO WEATHER HARVEST STATUS")
    print("=" * 60)
    print(f"Total Target Cells:    {total_cells:>6d}")
    print(f"Cached Cells (4,383d): {cached_count:>6d} ({cached_count / total_cells * 100:.1f}%)")
    print(f"Remaining Cells:       {remaining:>6d}")
    print(f"Failed Cells Logged:   {failed_count:>6d}")
    print(f"Projected Harvest ETA: {eta_days:>6.1f} days (at 127 cells/day)")
    print("-" * 60)
    print(f"Ledger 1m Consumed:    {fetcher.ledger.used(60.0):>6.1f} / {fetcher.ledger.minute_cap} weight")
    print(f"Ledger 1h Consumed:    {fetcher.ledger.used(3600.0):>6.1f} / {fetcher.ledger.hour_cap} weight")
    print(f"Ledger 24h Consumed:   {fetcher.ledger.used(86400.0):>6.1f} / {fetcher.ledger.day_cap} weight")
    print("=" * 60)


def run_dry_run(fetcher: WeatherFetcher) -> None:
    """Preview projected harvest cells, total weight, and pacing schedule."""
    cells = fetcher.get_cell_list()
    total_cells = len(cells)
    cached_count = sum(1 for c in cells if fetcher.is_cell_cached(str(c["cell_id"])))
    to_fetch = total_cells - cached_count

    projected_weight = to_fetch * CELL_WEIGHT
    projected_days = to_fetch / 127.0

    print("=" * 60)
    print("OPEN-METEO WEATHER HARVEST — DRY RUN PREVIEW")
    print("=" * 60)
    print(f"Total Grid Cells:      {total_cells:>6d}")
    print(f"Already Cached:        {cached_count:>6d}")
    print(f"Cells to Fetch:        {to_fetch:>6d}")
    print(f"Weight per Cell:       {CELL_WEIGHT:>6.2f} calls")
    print(f"Total Projected Weight:{projected_weight:>9.1f} calls")
    print(f"Estimated Duration:    {projected_days:>6.1f} days (at 127 cells/day, ~63 cells/hour)")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for weather fetcher."""
    parser = argparse.ArgumentParser(description="Open-Meteo Weather Harvester for Wildfire Early Alert")
    parser.add_argument("--max-minutes", type=float, default=125.0, help="Stop cleanly after this many minutes")
    parser.add_argument("--limit-cells", type=int, default=None, help="Maximum number of cells to process")
    parser.add_argument("--dry-run", action="store_true", help="Print harvest projections without fetching")
    parser.add_argument("--status", action="store_true", help="Print current caching and quota ledger status")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    fetcher = WeatherFetcher()

    if args.status:
        print_status(fetcher)
        return 0

    if args.dry_run:
        run_dry_run(fetcher)
        return 0

    if not acquire_lock():
        logger.error("Another fetch process holds %s. Exiting.", LOCK_FILE)
        return 4

    try:
        cells = fetcher.get_cell_list()
        start_session = time.time()
        max_seconds = args.max_minutes * 60.0
        fetched_count = 0

        for cell in cells:
            # Check elapsed time
            if time.time() - start_session >= max_seconds:
                logger.info("Reached session time limit of %.1f minutes. Stopping cleanly.", args.max_minutes)
                break

            # Check limit-cells
            if args.limit_cells is not None and fetched_count >= args.limit_cells:
                logger.info("Reached cell limit of %d. Stopping.", args.limit_cells)
                break

            cell_id = str(cell["cell_id"])
            if fetcher.is_cell_cached(cell_id):
                continue

            success, status, msg = fetcher.fetch_cell(cell)
            if not success:
                if status == 429:
                    logger.error("Rate limit encountered (%s). Halting session.", msg)
                    return 2
                logger.warning("Cell %s failed: %s. Continuing...", cell_id, msg)
            else:
                fetched_count += 1
                logger.info("Successfully fetched cell %s (%d fetched this session)", cell_id, fetched_count)

        logger.info("Session complete. Fetched %d cells.", fetched_count)
        return 0

    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
