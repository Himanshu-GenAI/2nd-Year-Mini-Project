import json
from pathlib import Path
from unittest.mock import patch

from src.data.quota import QuotaLedger

CELL_W = 62.61


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def test_fresh_ledger_allows_request(tmp_path: Path):
    ledger_file = tmp_path / "quota_ledger.json"
    clock = FakeClock()
    ledger = QuotaLedger(ledger_path=ledger_file, clock=clock)

    assert ledger.used(60.0) == 0.0
    assert ledger.used(3600.0) == 0.0
    assert ledger.used(86400.0) == 0.0
    assert ledger.seconds_until_allowed(CELL_W) == 0.0


def test_minute_cap_blocking_and_wait(tmp_path: Path):
    ledger_file = tmp_path / "quota_ledger.json"
    clock = FakeClock()
    ledger = QuotaLedger(ledger_path=ledger_file, clock=clock)

    # 7 requests * 62.61 = 438.27 <= 480
    for i in range(7):
        ledger.record(CELL_W, f"r00c{i:02d}", 200)

    assert ledger.used(60.0) == 7 * CELL_W
    # 8th request: 438.27 + 62.61 = 500.88 > 480
    wait = ledger.seconds_until_allowed(CELL_W)
    assert wait > 0.0
    # Wait should expire the first request after 60s
    clock.advance(wait + 0.1)
    assert ledger.seconds_until_allowed(CELL_W) == 0.0


def test_hourly_cap_blocks_after_63_requests(tmp_path: Path):
    ledger_file = tmp_path / "quota_ledger.json"
    clock = FakeClock()
    ledger = QuotaLedger(ledger_path=ledger_file, clock=clock)

    # Spread 63 requests evenly over 50 minutes (1 every ~45s)
    # Minute cap is never hit since requests are 45s apart
    for i in range(63):
        clock.advance(45.0)
        ledger.record(CELL_W, f"cell_{i}", 200)

    # 63 * 62.61 = 3944.43 <= 4000
    assert ledger.used(3600.0) == 63 * CELL_W
    # 64th request would be 4007.04 > 4000
    wait = ledger.seconds_until_allowed(CELL_W)
    assert wait > 0.0

    # Advance time past wait and verify it becomes allowed
    clock.advance(wait + 0.1)
    assert ledger.seconds_until_allowed(CELL_W) == 0.0


def test_daily_cap_blocks_after_127_requests(tmp_path: Path):
    ledger_file = tmp_path / "quota_ledger.json"
    clock = FakeClock()
    ledger = QuotaLedger(ledger_path=ledger_file, clock=clock)

    # Spread 127 requests across ~18 hours (~500s apart)
    # Neither minute nor hour cap will be triggered
    for i in range(127):
        clock.advance(500.0)
        ledger.record(CELL_W, f"cell_{i}", 200)

    # 127 * 62.61 = 7951.47 <= 8000
    assert ledger.used(86400.0) == 127 * CELL_W

    # 128th request: 7951.47 + 62.61 = 8014.08 > 8000
    wait = ledger.seconds_until_allowed(CELL_W)
    assert wait > 0.0

    # Advance time and verify
    clock.advance(wait + 0.1)
    assert ledger.seconds_until_allowed(CELL_W) == 0.0


def test_pruning_entries_older_than_24_hours(tmp_path: Path):
    ledger_file = tmp_path / "quota_ledger.json"
    clock = FakeClock()
    ledger = QuotaLedger(ledger_path=ledger_file, clock=clock)

    ledger.record(CELL_W, "old_cell", 200)
    assert len(ledger.entries) == 1

    # Advance 25 hours (90,000s)
    clock.advance(90_000.0)
    ledger.prune(clock())
    assert len(ledger.entries) == 0


def test_corrupted_or_missing_ledger_starts_empty(tmp_path: Path):
    missing_file = tmp_path / "missing.json"
    ledger1 = QuotaLedger(ledger_path=missing_file)
    assert ledger1.entries == []

    corrupted_file = tmp_path / "corrupted.json"
    corrupted_file.write_text("NOT_VALID_JSON{{{", encoding="utf-8")
    ledger2 = QuotaLedger(ledger_path=corrupted_file)
    assert ledger2.entries == []


def test_atomic_save_leaves_valid_file_on_exception(tmp_path: Path):
    ledger_file = tmp_path / "quota_ledger.json"
    clock = FakeClock()
    ledger = QuotaLedger(ledger_path=ledger_file, clock=clock)
    ledger.record(CELL_W, "cell_1", 200)

    assert ledger_file.exists()
    with open(ledger_file, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1

    # Simulate crash during os.replace
    with patch("os.replace", side_effect=OSError("Simulated disk error")):
        try:
            ledger.record(CELL_W, "cell_2", 200)
        except OSError:
            pass

    # Original file must remain intact and valid
    with open(ledger_file, encoding="utf-8") as f:
        recovered = json.load(f)
    assert len(recovered) == 1
    assert recovered[0]["cell_id"] == "cell_1"
