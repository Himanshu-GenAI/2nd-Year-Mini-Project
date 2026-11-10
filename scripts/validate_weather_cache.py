"""Validate full weather Parquet cache across all 1,167 grid cells."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DOCS_AUDIT_DIR,
    DOCS_DIR,
    MEAN_RAIN_YEARS,
    TMAX_MAX_C,
    TMAX_MIN_C,
    WEATHER_DIR,
    WEATHER_END,
    WEATHER_N_DAYS,
    WEATHER_START,
)
from src.data.grid import load_grid  # noqa: E402

VALIDATION_CSV_PATH = DOCS_AUDIT_DIR / "weather_validation.csv"
SUMMARY_MD_PATH = DOCS_DIR / "data_audit_weather_complete.md"


def validate_weather_cache() -> pd.DataFrame:
    """Validate every grid cell file in WEATHER_DIR and write audit outputs."""
    DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    grid_df = load_grid().sort_values("cell_id").reset_index(drop=True)
    expected_dates = pd.date_range(WEATHER_START, WEATHER_END, freq="D").date

    records = []
    for _, row in grid_df.iterrows():
        cid = str(row["cell_id"])
        f_path = WEATHER_DIR / f"{cid}.parquet"

        if not f_path.exists():
            records.append(
                {
                    "cell_id": cid,
                    "status": "missing",
                    "rows": 0,
                    "nulls": np.nan,
                    "mean_annual_rain_mm": np.nan,
                    "mean_tmax_c": np.nan,
                    "reason": "File missing",
                }
            )
            continue

        try:
            df = pd.read_parquet(f_path)
            n_rows = len(df)
            if n_rows != WEATHER_N_DAYS:
                records.append(
                    {
                        "cell_id": cid,
                        "status": "failed",
                        "rows": n_rows,
                        "nulls": np.nan,
                        "mean_annual_rain_mm": np.nan,
                        "mean_tmax_c": np.nan,
                        "reason": f"Row count {n_rows} != {WEATHER_N_DAYS}",
                    }
                )
                continue

            dates_series = pd.to_datetime(df["date"])
            if not (dates_series.dt.date.values == expected_dates).all():
                records.append(
                    {
                        "cell_id": cid,
                        "status": "failed",
                        "rows": n_rows,
                        "nulls": np.nan,
                        "mean_annual_rain_mm": np.nan,
                        "mean_tmax_c": np.nan,
                        "reason": "Date continuity mismatch",
                    }
                )
                continue

            n_nulls = int(df["temperature_2m_max"].isna().sum() + df["precipitation_sum"].isna().sum())
            if n_nulls > 0:
                records.append(
                    {
                        "cell_id": cid,
                        "status": "failed",
                        "rows": n_rows,
                        "nulls": n_nulls,
                        "mean_annual_rain_mm": np.nan,
                        "mean_tmax_c": np.nan,
                        "reason": f"{n_nulls} nulls found",
                    }
                )
                continue

            tmax_vals = df["temperature_2m_max"].values
            precip_vals = df["precipitation_sum"].values
            if (tmax_vals < TMAX_MIN_C).any() or (tmax_vals > TMAX_MAX_C).any() or (precip_vals < 0.0).any():
                records.append(
                    {
                        "cell_id": cid,
                        "status": "failed",
                        "rows": n_rows,
                        "nulls": 0,
                        "mean_annual_rain_mm": np.nan,
                        "mean_tmax_c": np.nan,
                        "reason": "Out of physical range",
                    }
                )
                continue

            # Compute distribution stats (train years 2015-2022 for mean annual rain)
            years = dates_series.dt.year
            rain_mask = (years >= MEAN_RAIN_YEARS[0]) & (years <= MEAN_RAIN_YEARS[1])
            n_years = float(MEAN_RAIN_YEARS[1] - MEAN_RAIN_YEARS[0] + 1)
            mean_annual_rain = float(precip_vals[rain_mask].sum() / n_years)
            mean_tmax = float(tmax_vals.mean())

            records.append(
                {
                    "cell_id": cid,
                    "status": "complete",
                    "rows": n_rows,
                    "nulls": 0,
                    "mean_annual_rain_mm": round(mean_annual_rain, 2),
                    "mean_tmax_c": round(mean_tmax, 2),
                    "reason": "OK",
                }
            )
        except Exception as e:
            records.append(
                {
                    "cell_id": cid,
                    "status": "failed",
                    "rows": 0,
                    "nulls": np.nan,
                    "mean_annual_rain_mm": np.nan,
                    "mean_tmax_c": np.nan,
                    "reason": str(e),
                }
            )

    val_df = pd.DataFrame(records)
    val_df.to_csv(VALIDATION_CSV_PATH, index=False)

    n_total = len(val_df)
    n_complete = int((val_df["status"] == "complete").sum())
    n_missing = int((val_df["status"] == "missing").sum())
    n_failed = int((val_df["status"] == "failed").sum())

    comp = val_df[val_df["status"] == "complete"]
    rain_stats = comp["mean_annual_rain_mm"].describe() if n_complete > 0 else None
    tmax_stats = comp["mean_tmax_c"].describe() if n_complete > 0 else None

    summary_lines = [
        "# Weather Cache Validation Audit",
        "",
        f"- **Total Grid Cells:** {n_total:,}",
        f"- **Cells Complete (4,383 valid rows):** {n_complete:,} ({n_complete / n_total * 100:.1f}%)",
        f"- **Cells Missing:** {n_missing:,}",
        f"- **Cells Failed:** {n_failed:,}",
        "",
    ]
    if n_complete > 0 and rain_stats is not None and tmax_stats is not None:
        summary_lines.extend(
            [
                "## Distribution Across Validated Cells",
                "",
                "| Metric | Min | 25% | Median | Mean | 75% | Max |",
                "|---|---|---|---|---|---|---|",
                (
                    f"| **Mean Annual Rainfall (mm, 2015–2022)** | {rain_stats['min']:.1f} | "
                    f"{rain_stats['25%']:.1f} | {rain_stats['50%']:.1f} | {rain_stats['mean']:.1f} | "
                    f"{rain_stats['75%']:.1f} | {rain_stats['max']:.1f} |"
                ),
                (
                    f"| **Mean Daily Tmax (°C, 2014–2025)** | {tmax_stats['min']:.2f} | "
                    f"{tmax_stats['25%']:.2f} | {tmax_stats['50%']:.2f} | {tmax_stats['mean']:.2f} | "
                    f"{tmax_stats['75%']:.2f} | {tmax_stats['max']:.2f} |"
                ),
                "",
            ]
        )

    SUMMARY_MD_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Weather Cache Validation: complete={n_complete}, missing={n_missing}, failed={n_failed} (total={n_total})")
    print(f"Saved validation CSV to {VALIDATION_CSV_PATH} and summary to {SUMMARY_MD_PATH}")
    return val_df


if __name__ == "__main__":
    validate_weather_cache()

