"""Verify downloaded weather cell Parquet files and quota ledger."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import WEATHER_DIR, WEATHER_END, WEATHER_N_DAYS, WEATHER_START  # noqa: E402

LEDGER_FILE = PROJECT_ROOT / "data" / "raw" / "weather" / "quota_ledger.json"


def verify_weather():
    parquet_files = sorted(WEATHER_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {WEATHER_DIR}:")
    for f in parquet_files:
        print(f"  - {f.name} ({f.stat().st_size} bytes)")

    assert len(parquet_files) >= 3, f"Expected at least 3 parquet files, found {len(parquet_files)}"

    for f in parquet_files[:3]:
        df = pd.read_parquet(f)
        print(f"\nVerifying {f.name}:")
        print(f"  Row count: {len(df)}")
        assert len(df) == WEATHER_N_DAYS, f"Expected {WEATHER_N_DAYS} rows, got {len(df)}"

        # Check required columns
        for col in ["date", "temperature_2m_max", "precipitation_sum", "cell_id", "fetched_at"]:
            assert col in df.columns, f"Missing column {col}"

        # Check nulls
        tmax_nulls = df["temperature_2m_max"].isnull().sum()
        precip_nulls = df["precipitation_sum"].isnull().sum()
        print(f"  Null counts: tmax={tmax_nulls}, precip={precip_nulls}")
        assert tmax_nulls == 0, f"Found nulls in temperature_2m_max in {f.name}"
        assert precip_nulls == 0, f"Found nulls in precipitation_sum in {f.name}"

        # Check dates
        min_date = str(df["date"].min())
        max_date = str(df["date"].max())
        print(f"  Date range: {min_date} to {max_date}")
        assert min_date == WEATHER_START, f"Min date mismatch: {min_date} vs {WEATHER_START}"
        assert max_date == WEATHER_END, f"Max date mismatch: {max_date} vs {WEATHER_END}"

        # Check ranges
        tmax_min = df["temperature_2m_max"].min()
        tmax_max = df["temperature_2m_max"].max()
        precip_min = df["precipitation_sum"].min()
        precip_max = df["precipitation_sum"].max()
        print(f"  Tmax range: [{tmax_min:.2f}, {tmax_max:.2f}] °C")
        print(f"  Precip range: [{precip_min:.2f}, {precip_max:.2f}] mm")
        assert -20.0 <= tmax_min and tmax_max <= 60.0, "Tmax outside [-20, 60] range"
        assert precip_min >= 0.0, "Precipitation is negative"

    # Verify ledger
    assert LEDGER_FILE.exists(), "Quota ledger file does not exist"
    with open(LEDGER_FILE, encoding="utf-8") as lf:
        entries = json.load(lf)
    print(f"\nQuota Ledger Entries ({len(entries)} total):")
    for e in entries:
        print(f"  - Cell: {e['cell_id']}, weight: {e['weight']:.2f}, status: {e['status']}")
        assert e["status"] == 200, f"HTTP status was not 200: {e}"
        assert abs(e["weight"] - 62.61) < 0.01, f"Weight not ~62.61: {e['weight']}"

    print("\nALL 3 CELLS & LEDGER VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    verify_weather()
