# Week 1 Data Audit Note: Wildfire Early Alert (PS-113)

**Date:** 2026-10-10 | **Scope:** India MODIS FIRMS (2015–2025), Open-Meteo Weather, 0.5° Grid

---

## 1. Data Summary Table

| Dataset | Raw Count | Usable Count | Temporal Span | Spatial Domain | Primary File |
|---|---|---|---|---|---|
| **NASA FIRMS MODIS** | 923,584 | 834,958 (`type==0`, `conf>=30`) | 2015–2025 | India Box | `firms_audit.csv` |
| **Open-Meteo Weather** | 4,383 days/cell | 4,383 days/cell (0 nulls) | 2014–2025 | 1,167 Cells | `weather_probe.csv` |
| **DataMeet Boundaries** | 36 States/UTs | 1 Dissolved Poly (3.27M km²) | 2026-10-10 | National | `grid_summary.csv` |
| **Candidate Grid** | 3,968 cells | 1,167 kept cells (≥50% share) | 11-Year Panel | Lat [6,38], Lon [67,98] | `grid_summary.csv` |

---

## 2. Quality Flags

1. **Synopsis Discrepancy:** Raw FIRMS rows count 923,584 (vs. 1.88M synopsis claim; ratio 0.4913; unverified estimate).
2. **Sensor Authenticity:** 100% MODIS (Terra/Aqua); 0 NRT records found.
3. **Completeness:** All 11 years (2015–2025) and all 132 months contain active detections (0 zero-row months).
4. **Fire Type & Confidence:** `type` exists; `type==0` keeps 834,958 rows (94.40% of conf≥30). `conf>=30` keeps 884,467 rows (95.76% of dedup).

---

## 3. Weather Feasibility (Recomputed)

1. **Continuity:** Single-cell probe (30.25°N, 78.75°E) returned 4,383 continuous daily records with 0 nulls for 2014–2025 (365 valid spin-up days in 2014).
2. **Fractional Call Weight:** Open-Meteo formula `weight = nLocations * (nDays / 14) * (nVariables / 10)` yields 187.84 calls for single-cell probe and 3,130.71 calls for ramp batch 50 (triggering HTTP 429 minutely throttling).
3. **Stop Gate 2 FAIL:** For 1,167 cells × 4,383 days, total weight is 73,065.87 calls (2 vars) and 219,209.28 calls (6 vars). At 80% daily cap (8,000 calls/day), harvesting requires **9.1 days** (2 vars) or **27.4 days** (6 vars). Exceeding 3 days triggers **Stop Gate 2 FAIL**.

---

## 4. Grid and Coverage

1. **Lattice:** 64 rows × 62 cols = 3,968 candidate cells (0.5° resolution).
2. **Inclusion:** 1,167 cells have `inside_share >= 0.50` (EPSG:6933), covering 3,270,089.99 km² national land area.
3. **Detection Retention:** Captures 811,280 detections (97.16% of filtered total, 97.33% of in-India).
4. **Dropped Detections:** 22,233 detections (2.67% of in-India) fall in dropped border cells (<10% threshold).
5. **Panel Volume:** 1,167 cells × 4,018 study days = 4,689,006 cell-days. Spin-up weather = 5,114,961 cell-days.

---

## 5. Frozen and Reopened Decisions

- **D1:** Retain `type == 0` (presumed vegetation fires; 834,958 detections).
- **D2:** Confidence threshold `confidence >= 30` (884,467 detections).
- **D3:** Spatial 0.5° grid with `inside_share >= 0.50` (1,167 cells).
- **D4:** Labels: `new_fire_next_48h` and `fire_next_48h`.
- **D5:** Splits: Train (2015–2022), Val (2023), Test (2024–2025) with 2-day gaps.
- **D6 (Reopened):** Weather harvesting plan (Stop Gate 2 FAIL; under review).
- **D7:** Deduplication on `(lat, lon, acq_date, acq_time, satellite)`.
- **D8:** Day boundary anchored to UTC `acq_date` and Open-Meteo `timezone=GMT`.
- **D9:** MODIS Terra and Aqua combined.
- **D10:** Detections outside kept cells (25,508 total) excluded from panel.
- **D11:** KBDI rainfall from 2015–2022 only; spin-up from 2014-01-01.

---

## 6. Open Questions

1. **Weather Route (D6):** Choice between spatial sampling (300 cells in 2.3 days), Copernicus CDS bulk download, or 9–10 day phased fetch.
2. **Diurnal Alignment (DEF-1):** UTC vs. IST (+5:30) day boundary (Week 2).
3. **Island Gridding (DEF-2):** Andaman & Lakshadweep sub-0.5° dashboard coverage (Week 4).

---

## 7. Risk Flags for Weeks 2–3

1. **Severe Imbalance:** Positive fire cell-days comprise <2% of 4.69M cell-days (*evidence:* `grid_summary.csv`). Requires PR-AUC and Brier score over ROC-AUC.
2. **Weather API Throttling:** Quota restricts throughput to 127 cells/day (2 vars) or 42 cells/day (6 vars) (*evidence:* `weather_probe.csv`). Requires single-cell batching with 10–30s delay.

---

## 8. Number Traceability Table

| Value | Description | Source File in `docs/audit/` |
|---|---|---|
| **923,584** | Raw FIRMS detections | `firms_audit.csv` |
| **884,467** | Detections with `confidence >= 30` | `firms_audit.csv` |
| **834,958** | Detections with `type == 0` and `conf >= 30` | `firms_audit.csv`, `grid_coverage.csv` |
| **0.4913** | Ratio of raw rows to 1.88M synopsis claim | `firms_audit.csv` |
| **81,615.0** | Median annual raw detections | `firms_yearly_summary.csv` |
| **132** | Active calendar months (2015–2025) | `firms_audit.csv` |
| **4,383** | Daily weather records per cell | `weather_probe.csv`, `grid_summary.csv` |
| **365** | Valid 2014 spin-up weather days | `weather_probe.csv` |
| **187.84** | Single-cell probe call weight | `weather_probe.csv` |
| **3,130.71** | Ramp batch 50 call weight (429 error) | `weather_probe.csv` |
| **73,065.87** | Total projected weight (1,167 cells, 2 vars) | `weather_probe.csv` |
| **219,209.28** | Total projected weight (1,167 cells, 6 vars) | `weather_probe.csv` |
| **9.1** | Days needed for 2 vars (80% cap) | `weather_probe.csv` |
| **27.4** | Days needed for 6 vars (80% cap) | `weather_probe.csv` |
| **3,968** | Candidate 0.5° cells (64x62) | `grid_summary.csv` |
| **1,167** | Kept cells (`inside_share >= 0.50`) | `grid_summary.csv`, `grid_coverage.csv` |
| **3,270,089.99** | India dissolved land area (km²) | `grid_summary.csv` |
| **811,280** | Detections in kept cells | `grid_coverage.csv` |
| **97.16%** | Share of detections in kept cells | `grid_coverage.csv` |
| **22,233** | In-India detections in dropped cells (2.67%) | `grid_coverage.csv` |
| **3,275** | Detections outside India boundary (0.39%) | `grid_coverage.csv` |
| **4,689,006** | Total study ML cell-days | `grid_summary.csv` |
| **5,114,961** | Total weather spin-up rows | `grid_summary.csv` |
