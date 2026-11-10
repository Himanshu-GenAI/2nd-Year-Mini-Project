# Copernicus ERA5 Bulk Ingestion Feasibility Note

**Date:** 2026-11-10 | **Scope:** Alternative Meteorological Sourcing & Extended Variables (RH, Wind)

---

## 1. Dataset & Spatial Grid
- **Dataset:** `derived-era5-single-levels-daily-statistics` on the Copernicus Climate Data Store (CDS) provides daily statistics (`daily_maximum` for `2m_temperature`, `daily_sum` for `total_precipitation`) on a native **0.25° × 0.25°** regular latitude-longitude grid.

## 2. Account, License & Request Limits
- **Account & License:** Requires registering a free CDS account, configuring a `cdsapi` API token (`~/.cdsapirc`), and **explicitly accepting the CC-BY 4.0 / Copernicus License Terms** on the web dataset page before programmatic API requests are authorized.
- **Request Limits:** CDS restricts request volume by total field items ($\text{variables} \times \text{days}$); ECMWF recommends chunking multi-year queries to **1 month per API request** restricted by bounding box (`area: [38, 67, 6, 98]`) to avoid `"cost limits exceeded"` errors.
- **Queue Time:** Unlike pre-archived hourly files, daily statistics are computed dynamically on-the-fly from hourly ERA5 archives. Queue wait times depend heavily on global load—ranging from minutes to several hours (**exact SLA unknown/variable**).

## 3. Adding Humidity & Wind Variables
- Request `2m_dewpoint_temperature` (to derive Relative Humidity alongside `2m_temperature` via the Magnus-Tetens formula), plus `10m_u_component_of_wind` and `10m_v_component_of_wind` (to compute daily wind speed $\sqrt{u^2 + v^2}$).
- Spatial alignment requires aggregating the 0.25° ERA5 cells (4 sub-cells per 0.5° cell) to match our 1,167 kept cells.

## 4. Recommendation for Model v1 (Decision D17)
- **No for Model v1.** Retrieving 12 years (2014–2025) in 1-month chunks requires ~144 separate queued CDS jobs plus NetCDF-to-Parquet regridding pipelines, adding operational risk without an immediate cheap path. Model v1 proceeds with our frozen Open-Meteo variables (`tmax`, `precip`, `kbdi`), deferring humidity/wind to post-v1 experiments.

