from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.config import GRID_PATH, WEATHER_N_DAYS, WEATHER_START
from src.data.quota import QuotaLedger
from src.data.weather import WeatherFetcher, acquire_lock, main, release_lock


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, s: float) -> None:
        self.current += s


def make_valid_canned_payload() -> dict:
    start_d = date.fromisoformat(WEATHER_START)
    times = [(start_d + timedelta(days=i)).isoformat() for i in range(WEATHER_N_DAYS)]
    tmax = [25.0] * WEATHER_N_DAYS
    precip = [0.0] * WEATHER_N_DAYS
    return {
        "daily": {
            "time": times,
            "temperature_2m_max": tmax,
            "precipitation_sum": precip,
        }
    }


def make_mock_grid(tmp_path: Path) -> Path:
    grid_file = tmp_path / "mock_grid.parquet"
    df = pd.DataFrame(
        {
            "cell_id": [f"r{r:02d}c{c:02d}" for r in range(5) for c in range(5)],
            "lat_c": [10.0 + r for r in range(5) for _ in range(5)],
            "lon_c": [70.0 + c for _ in range(5) for c in range(5)],
        }
    )
    df.to_parquet(grid_file, index=False)
    return grid_file


def test_deterministic_shuffle(tmp_path: Path):
    grid_file = make_mock_grid(tmp_path)
    fetcher1 = WeatherFetcher(grid_path=grid_file, weather_dir=tmp_path / "weather")
    fetcher2 = WeatherFetcher(grid_path=grid_file, weather_dir=tmp_path / "weather")

    cells1 = fetcher1.get_cell_list()
    cells2 = fetcher2.get_cell_list()

    assert len(cells1) == len(cells2) == 25
    assert [c["cell_id"] for c in cells1] == [c["cell_id"] for c in cells2]


def test_skip_complete_file(tmp_path: Path):
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir(parents=True)
    fetcher = WeatherFetcher(weather_dir=weather_dir)

    cell_file = weather_dir / "r01c01.parquet"
    df = pd.DataFrame({"dummy": range(WEATHER_N_DAYS)})
    df.to_parquet(cell_file)

    assert fetcher.is_cell_cached("r01c01") is True


def test_delete_and_refetch_wrong_row_count(tmp_path: Path):
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir(parents=True)
    fetcher = WeatherFetcher(weather_dir=weather_dir)

    cell_file = weather_dir / "r01c01.parquet"
    # Wrong row count
    df = pd.DataFrame({"dummy": range(100)})
    df.to_parquet(cell_file)

    assert fetcher.is_cell_cached("r01c01") is False
    assert not cell_file.exists()


def test_validation_rejects_nulls(tmp_path: Path):
    fetcher = WeatherFetcher(weather_dir=tmp_path / "weather")
    payload = make_valid_canned_payload()
    payload["daily"]["temperature_2m_max"][10] = None

    with pytest.raises(ValueError, match="null values"):
        fetcher.validate_payload(payload)


def test_validation_rejects_wrong_length(tmp_path: Path):
    fetcher = WeatherFetcher(weather_dir=tmp_path / "weather")
    payload = make_valid_canned_payload()
    payload["daily"]["time"] = payload["daily"]["time"][:-1]

    with pytest.raises(ValueError, match="mismatch"):
        fetcher.validate_payload(payload)


def test_validation_rejects_date_gap(tmp_path: Path):
    fetcher = WeatherFetcher(weather_dir=tmp_path / "weather")
    payload = make_valid_canned_payload()
    payload["daily"]["time"][50] = "2020-01-01"

    with pytest.raises(ValueError, match="not continuous"):
        fetcher.validate_payload(payload)


def test_validation_rejects_out_of_range_temp(tmp_path: Path):
    fetcher = WeatherFetcher(weather_dir=tmp_path / "weather")
    payload = make_valid_canned_payload()
    payload["daily"]["temperature_2m_max"][5] = 75.0  # Above 60 C

    with pytest.raises(ValueError, match="out of valid range"):
        fetcher.validate_payload(payload)


def test_failed_cell_leaves_no_final_file(tmp_path: Path):
    weather_dir = tmp_path / "weather"
    weather_dir.mkdir(parents=True)
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_session.get.return_value = mock_resp
    test_ledger = QuotaLedger(ledger_path=tmp_path / "test_ledger.json")

    fetcher = WeatherFetcher(
        weather_dir=weather_dir,
        ledger=test_ledger,
        session=mock_session,
        sleep_fn=lambda _: None,
        failed_log=tmp_path / "failed.csv",
        log_file=tmp_path / "fetch.log",
    )
    success, status, _ = fetcher.fetch_cell({"cell_id": "r99c99", "lat_c": 20.0, "lon_c": 75.0})

    assert success is False
    assert status == 500
    assert not (weather_dir / "r99c99.parquet").exists()


def test_rate_limit_429_stops_with_retries(tmp_path: Path):
    weather_dir = tmp_path / "weather"
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Too Many Requests"
    mock_session.get.return_value = mock_resp
    test_ledger = QuotaLedger(ledger_path=tmp_path / "test_ledger.json")

    fetcher = WeatherFetcher(
        weather_dir=weather_dir,
        ledger=test_ledger,
        session=mock_session,
        sleep_fn=lambda _: None,
        failed_log=tmp_path / "failed.csv",
        log_file=tmp_path / "fetch.log",
    )
    success, status, msg = fetcher.fetch_cell({"cell_id": "r99c99", "lat_c": 20.0, "lon_c": 75.0})

    assert success is False
    assert status == 429
    assert "Exceeded max 429 retries" in msg
    # Should have retried 3 times (4 requests total)
    assert mock_session.get.call_count == 4


def test_daily_rate_limit_stops_immediately(tmp_path: Path):
    weather_dir = tmp_path / "weather"
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Daily limit of requests exceeded"
    mock_session.get.return_value = mock_resp
    test_ledger = QuotaLedger(ledger_path=tmp_path / "test_ledger.json")

    fetcher = WeatherFetcher(
        weather_dir=weather_dir,
        ledger=test_ledger,
        session=mock_session,
        sleep_fn=lambda _: None,
        failed_log=tmp_path / "failed.csv",
        log_file=tmp_path / "fetch.log",
    )
    success, status, msg = fetcher.fetch_cell({"cell_id": "r99c99", "lat_c": 20.0, "lon_c": 75.0})

    assert success is False
    assert status == 429
    assert "Daily limit exceeded" in msg
    # Stops immediately without retries
    assert mock_session.get.call_count == 1


def test_lock_refuses_second_instance(tmp_path: Path):
    test_lock = tmp_path / "fetch.lock"
    release_lock(test_lock)
    assert acquire_lock(test_lock) is True
    # Second acquisition fails while process is alive
    assert acquire_lock(test_lock) is False
    release_lock(test_lock)
    assert acquire_lock(test_lock) is True
    release_lock(test_lock)


@pytest.mark.skipif(not GRID_PATH.exists(), reason="Grid file not available in environment")
def test_dry_run_makes_no_requests(capsys):
    ret = main(["--dry-run"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "DRY RUN PREVIEW" in captured.out
    assert "Total Grid Cells:        1167" in captured.out
    assert "Cells to Fetch:" in captured.out
    assert "Weight per Cell:" in captured.out
