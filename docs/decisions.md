# Wildfire Early Alert (PS-113) - Decision Log

**Last Updated:** 2026-10-10  
**Project:** Wildfire Early Alert System for India (2015–2025)

---

## 1. Frozen Decisions (Week 1)

| ID | Date | Decision | Rationale | Evidence File | Owner | Status |
|---|---|---|---|---|---|---|
| **D1** | 2026-10-10 | Fire-type filter: retain `type == 0` (presumed vegetation fires). | Phase 1 audit verdict is `EXISTS` (94.40% retained). Non-zero codes (2, 3) represent static industrial/other thermal anomalies. | `docs/data_audit_firms.md` | Team | `frozen` |
| **D2** | 2026-10-10 | Detection confidence threshold set to `confidence >= 30`. | Retains 884,467 / 923,584 records (95.76%), eliminating low-confidence sensor noise while preserving valid ignitions. | `docs/data_audit_firms.md` | Team | `frozen` |
| **D3** | 2026-10-10 | Spatial grid: 0.5° resolution over lat [6.0, 38.0], lon [67.0, 98.0] (3,968 candidate cells). Inclusion rule `inside_share >= 0.50` (EPSG:6933) keeps 1,167 cells capturing 97.16% of total filtered detections. | Measured cell count is 1,167. Only 2.67% of in-India detections lost in dropped border cells (well within <10% ceiling). | `docs/audit/grid_summary.csv` | Team | `frozen` |
| **D4** | 2026-10-10 | Labels: `new_fire_next_48h(c,t) = 1` if cell has ≥1 kept detection on UTC dates t+1 or t+2 and none on t-2, t-1, t; primary evaluation set contains only cell-days with no detection on t-2..t; `fire_next_48h(c,t) = 1` if ≥1 detection on t+1 or t+2 (all cell-days). | Standardizes operational early warning for new ignitions versus continuous burn monitoring. | `src/config.py` | Team | `frozen` |
| **D5** | 2026-10-10 | Temporal splits: Train 2015-01-03 to 2022-12-29; Val 2023-01-01 to 2023-12-29; Test 2024-01-01 to 2025-12-29; with 2-day buffer gaps (Dec 30–31) between splits. | Prevents temporal leakage given the 2-day forward label horizon and 3-day lookback window. | `src/config.py`, `tests/test_config.py` | Team | `frozen` |
| **D6** | 2026-10-10 | Weather harvesting: Open-Meteo Archive API. Recomputed with fractional call weighting `weight = nLocations * (nDays/14) * (nVariables/10)`. Total weight is 73,066 (2 vars) or 219,209 (6 vars), requiring 9.1 to 27.4 days at 80% limit cap. | Stop Gate 2 triggered FAIL (>3 days). Compliant fetch plan requires user selection between spatial sampling, temporal reduction, Copernicus CDS bulk download, or multi-day phased background ingestion. | `docs/data_audit_weather.md`, `docs/audit/weather_probe.csv` | Team | `reopened / pending` |
| **D7** | 2026-10-10 | Deduplication: drop duplicates on `(latitude, longitude, acq_date, acq_time, satellite)`, keeping first. | Eliminates duplicate telemetry across archive files. | `docs/data_audit_firms.md` | Team | `frozen` |
| **D8** | 2026-10-10 | Day boundary definition: UTC `acq_date` and Open-Meteo `timezone=GMT`. | Ensures consistent temporal alignment across disparate global satellite and meteorological datasets. | `src/config.py`, `docs/data_audit_weather.md` | Team | `frozen` |
| **D9** | 2026-10-10 | Satellite sensor fusion: combine MODIS Terra (morning pass) and Aqua (afternoon pass) detections. | Maximizes temporal sampling of diurnal wildfire cycles. | `docs/data_audit_firms.md` | Team | `frozen` |
| **D10** | 2026-10-10 | Out-of-grid handling: detections outside kept cells (3,275 outside boundary, 22,233 in dropped border cells) are excluded from ML panel and documented. | Maintains strict spatial boundary consistency across panel matrix. | `docs/audit/grid_coverage.csv` | Team | `frozen` |
| **D11** | 2026-10-10 | KBDI parameters: mean annual precipitation calculated from training period (2015–2022) only; drought index recursive spin-up starts 2014-01-01. | Eliminates lookahead leakage from test/validation rainfall into historical drought calculations. | `src/config.py` | Team | `frozen` |

---

## 2. External Data Source Provenance

- **India State Boundaries:** DataMeet Maps Repository (`States/Admin2.shp`, `Admin2.shx`, `Admin2.dbf`, `Admin2.prj`, `Admin2.cpg`)
  - Source URL: `https://raw.githubusercontent.com/datameet/maps/master/States/`
  - Download Date: 2026-10-10
  - Projected National Area (EPSG:6933): 3,270,089.99 km² (36 States & UTs)
- **NASA FIRMS MODIS C6.1:** `data/raw/firms/fire_archive_M-C61_808434.csv` (923,584 raw detections, 2015–2025)
- **Open-Meteo Historical Weather:** `https://archive-api.open-meteo.com/v1/archive` (ERA5/ERA5-Land reanalysis, 2014–2025)

---

## 3. Deferred Decisions

| ID | Open Question | Reason Deferred | Owner | Target Week |
|---|---|---|---|---|
| **DEF-1** | Local IST Day Boundary (+5:30 offset) vs UTC | Requires analyzing diurnal FIRMS detection distribution against ERA5 hourly reanalysis to confirm if local solar day reduces classification error. | Team | Week 2 |
| **DEF-2** | Island Territory Dedicated Sub-Grids (Andaman & Lakshadweep) | Island cells have `inside_share < 0.50` at 0.5° resolution; regional fine-mesh subgridding deferred to operational dashboard phase. | Team | Week 4 |
| **DEF-3** | VIIRS 375m High-Resolution Sensor Fusion | Initial ML baseline is calibrated on 11-year MODIS archive; VIIRS integration evaluated as extension deliverable. | Team | Post-Week 4 |
