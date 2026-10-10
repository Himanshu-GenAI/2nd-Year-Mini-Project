# Week 1 Data Audit Note: Wildfire Early Alert (PS-113)

**Date:** 2026-10-10 | **Scope:** India MODIS FIRMS (2015–2025), Open-Meteo Weather, 0.5° Spatial Grid

---

## 1. Data Summary Table

| Dataset | Raw Count | Filtered / Usable Count | Temporal Span | Spatial Domain | Primary File |
|---|---|---|---|---|---|
| **NASA FIRMS MODIS C6.1** | 923,584 | 834,958 (`type==0`, `conf>=30`) | 2015-01-01 to 2025-12-31 | India Footprint | `firms_audit.csv` |
| **Open-Meteo Weather** | 4,383 days/cell | 4,383 days/cell (0 nulls) | 2014-01-01 to 2025-12-31 | 1,167 Grid Cells | `weather_probe.csv` |
| **DataMeet Boundaries** | 36 States/UTs | 1 Dissolved Poly (3.27M km²) | 2026-10-10 Snapshot | India National | `grid_summary.csv` |
| **Candidate Grid** | 3,968 cells | 1,167 kept cells (≥50% share) | 11-Year Panel | Lat [6,38], Lon [67,98] | `grid_summary.csv` |

---

## 2. Quality Flags

1. **Synopsis Discrepancy:** Raw FIRMS rows count 923,584 (vs. 1.88M synopsis claim; ratio 0.4913). The synopsis figure was an unverified estimate.
2. **Sensor Authenticity:** 100% MODIS (Terra/Aqua). 0 NRT records found.
3. **Completeness:** All 11 years (2015–2025) and all 132 calendar months contain active detections (0 zero-row months).
4. **Fire Type & Confidence:** `type` column exists across all records. Retaining `type==0` keeps 834,958 rows (94.40% of conf≥30). `conf>=30` keeps 884,467 rows (95.76% of dedup).

---

## 3. Weather Feasibility

1. **Spin-up & Continuity:** Single-cell probe (lat 30.25, lon 78.75) returned 4,383 continuous daily records with 0 nulls for 2014–2025. 2014 spin-up is fully intact (365 valid days).
2. **Variables:** Verified daily availability of `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`, and `relative_humidity_2m_mean`.
3. **Quota Compliance:** Multi-cell batch harvesting of all 1,167 cells at batch size 20 requires 59 requests and 1,167 weighted calls. Total fetch completes in ~8 minutes, consuming 11.7% of the 10,000 daily free-tier limit (<1 day required).

---

## 4. Grid and Coverage

1. **Candidate Lattice:** 64 rows × 62 cols = 3,968 candidate cells (0.5° resolution).
2. **Inclusion Metric:** 1,167 cells have `inside_share >= 0.50` in EPSG:6933 equal-area projection, covering 3,270,089.99 km² national land area.
3. **Detection Retention:** Captures 811,280 detections (97.16% of all filtered detections, 97.33% of in-India detections).
4. **Dropped In-India Detections:** 22,233 detections (2.67% of in-India) occur in dropped peripheral/island cells (well under the 10.0% stop threshold).
5. **Panel Volume:** 1,167 cells × 4,018 study days = 4,689,006 ML cell-days. Weather spin-up = 5,114,961 cell-days.

---

## 5. Frozen Decisions

- **D1:** Keep `type == 0` (presumed vegetation fires; 834,958 detections).
- **D2:** Confidence threshold set to `confidence >= 30` (884,467 detections).
- **D3:** Spatial 0.5° grid with `inside_share >= 0.50` keeping 1,167 cells.
- **D4:** Labels: `new_fire_next_48h` (t+1..t+2 detection with no t-2..t fire) and `fire_next_48h`.
- **D5:** Temporal splits: Train (2015–2022), Val (2023), Test (2024–2025) with 2-day buffer gaps.
- **D6:** Open-Meteo weather variables (6 channels), batch size 20, 5s pause.
- **D7:** Deduplication on `(lat, lon, acq_date, acq_time, satellite)`.
- **D8:** Day boundary anchored to UTC `acq_date` and Open-Meteo `timezone=GMT`.
- **D9:** MODIS Terra and Aqua detections combined.
- **D10:** Detections outside kept cells (25,508 total) excluded from panel and audited.
- **D11:** KBDI annual rainfall computed from 2015–2022 only; spin-up from 2014-01-01.

---

## 6. Open Questions

1. **Diurnal Solar vs. UTC Alignment (DEF-1):** Whether converting UTC day to IST (+5:30) improves alignment with MODIS overpasses without complicating ERA5 daily rollups (Week 2).
2. **Island Gridding (DEF-2):** Andaman & Lakshadweep sub-0.5° representation for regional dashboards (Week 4).

---

## 7. Risk Flags for Weeks 2–3

1. **Severe Class Imbalance:** Positive fire cell-days comprise <2% of 4.69M cell-days. *Evidence:* 811,280 detections across 4,689,006 cell-days (`grid_summary.csv`). Requires PR-AUC / calibrated Brier scoring over ROC-AUC.
2. **Open-Meteo Minutely Throttling:** Rapid bursts (>600 calls/min or >20 bulk requests/min) trigger HTTP 429. *Evidence:* `weather_probe.csv` (batch 50 hit minutely burst limit). Requires strict 5s delays.

---

## 8. Number Traceability Table

| Metric Value in Note | Description | Source File in `docs/audit/` |
|---|---|---|
| **923,584** | Total raw FIRMS detections | `firms_audit.csv`, `firms_yearly_summary.csv` |
| **884,467** | Detections with `confidence >= 30` | `firms_audit.csv` |
| **834,958** | Detections with `type == 0` and `conf >= 30` | `firms_audit.csv`, `grid_coverage.csv` |
| **0.4913** | Ratio of raw rows to 1.88M synopsis claim | `firms_audit.csv` |
| **81,615.0** | Median annual raw detections | `firms_yearly_summary.csv` |
| **132** | Total active calendar months (2015–2025) | `firms_audit.csv` |
| **4,383** | Daily weather records per cell (2014–2025) | `weather_probe.csv`, `grid_summary.csv` |
| **365** | Valid 2014 spin-up weather days | `weather_probe.csv` |
| **10,000** | Free daily API call quota | `weather_probe.csv` |
| **3,968** | Total 0.5° candidate cells (64x62) | `grid_summary.csv` |
| **1,167** | Kept cells with `inside_share >= 0.50` | `grid_summary.csv`, `grid_coverage.csv` |
| **3,270,089.99** | India dissolved land area (km²) | `grid_summary.csv` |
| **811,280** | Filtered detections inside kept cells | `grid_coverage.csv` |
| **97.16%** | Share of total detections in kept cells | `grid_coverage.csv` |
| **22,233** | In-India detections in dropped cells (2.67%) | `grid_coverage.csv` |
| **3,275** | Detections outside India boundary (0.39%) | `grid_coverage.csv` |
| **4,689,006** | Total study ML cell-days (1,167 x 4,018) | `grid_summary.csv` |
| **5,114,961** | Total weather spin-up rows (1,167 x 4,383) | `grid_summary.csv` |
| **59** | Total HTTP requests for weather harvesting | `weather_probe.csv` |
