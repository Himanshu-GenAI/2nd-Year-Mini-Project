"""KBDI Validation & Demonstration Script.

Loads cached weather cells, computes daily Keetch-Byram Drought Index (KBDI),
and plots the multi-year trajectory against precipitation to verify seasonal dynamics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import WEATHER_DIR  # noqa: E402
from src.features.kbdi import compute_kbdi  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_demo():
    cached_files = sorted(WEATHER_DIR.glob("*.parquet"))
    if not cached_files:
        print("No cached weather parquet files found. Using synthetic probe.")
        return

    # Load first cached cell
    target_file = cached_files[0]
    cell_id = target_file.stem
    df = pd.read_parquet(target_file)
    print(f"Loaded cell {cell_id} with {len(df)} records ({df['date'].min()} to {df['date'].max()})")

    # Filter to 2015-2022 for clean demonstration
    df["dt"] = pd.to_datetime(df["date"])
    mask = (df["dt"].dt.year >= 2015) & (df["dt"].dt.year <= 2022)
    demo_df = df[mask].sort_values("dt").reset_index(drop=True)

    tmax_arr = demo_df["temperature_2m_max"].values.reshape(-1, 1)
    precip_arr = demo_df["precipitation_sum"].values.reshape(-1, 1)

    # Compute mean annual rainfall for this cell over 2015-2022
    total_rain = demo_df["precipitation_sum"].sum()
    mean_annual_mm = np.array([total_rain / 8.0])
    print(f"Cell {cell_id} Mean Annual Rainfall: {mean_annual_mm[0]:.1f} mm/year")

    # Compute KBDI
    kbdi_vals = compute_kbdi(tmax_arr, precip_arr, mean_annual_mm, q0=0.0)[:, 0]
    demo_df["kbdi"] = kbdi_vals

    print(f"KBDI Summary: min={kbdi_vals.min():.1f}, max={kbdi_vals.max():.1f}, mean={kbdi_vals.mean():.1f}")
    assert 0.0 <= kbdi_vals.min() and kbdi_vals.max() <= 800.0, "KBDI out of [0, 800] range"

    # Plot figure: Top subplot KBDI, bottom subplot daily precipitation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(demo_df["dt"], demo_df["kbdi"], color="#d62728", lw=1.2, label="KBDI (0 - 800)")
    ax1.axhline(200, color="green", linestyle=":", alpha=0.7, label="Low Fire Hazard (< 200)")
    ax1.axhline(600, color="orange", linestyle=":", alpha=0.7, label="High Fire Hazard (> 600)")
    ax1.set_ylabel("KBDI (Drought Index)")
    ax1.set_title(f"Keetch-Byram Drought Index (KBDI) Trajectory — Cell {cell_id} (2015–2022)")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right")
    ax1.set_ylim(-10, 810)

    ax2.bar(demo_df["dt"], demo_df["precipitation_sum"], color="#1f77b4", width=1.5, label="Daily Rain (mm)")
    ax2.set_ylabel("Precipitation (mm)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = FIGURES_DIR / "kbdi_demo.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved KBDI demonstration plot to {plot_path}")


if __name__ == "__main__":
    run_demo()
