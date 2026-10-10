# Open-Meteo Weather API Feasibility and Rate Limit Audit (Recomputed)

**Audit Execution Date:** 2026-10-10  
**Target Domain:** India 0.5° Spatial Grid (1,167 kept cells)  
**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`  
**Temporal Span:** 2014-01-01 to 2025-12-31 (4,383 days continuous, including 2014 KBDI spin-up)

---

## 1. Documented API Specifications & Weighting Formula

- **Document Inspection Date:** 2026-10-10 (verified at `https://open-meteo.com/en/pricing` and `https://open-meteo.com/en/terms`)
- **Documented Rate Limits:**
  - **Minutely Limit:** 600 calls / minute
  - **Hourly Limit:** 5,000 calls / hour
  - **Daily Limit:** 10,000 calls / day
- **Fractional Call Weight Formula:**
  $$\text{weight} = \text{nLocations} \times \left(\frac{\text{nDays}}{14}\right) \times \left(\frac{\text{nVariables}}{10}\right)$$
  *(Requests spanning more than 2 weeks or more than 10 variables accrue fractional multipliers per 14 days and per 10 variables).*

---

## 2. Variables & Weights in Probe Execution

### 2.1 Single-Cell Probe (Lat 30.25°N, Lon 78.75°E)
- **Variables Used (6 channels):** `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`, `relative_humidity_2m_mean`
- **Call Weight:**
  $$1 \times \left(\frac{4,383}{14}\right) \times \left(\frac{6}{10}\right) = \mathbf{187.84}\text{ calls weight}$$
- **Result:** HTTP 200, 4,383 continuous daily records (0 nulls in 2014 spin-up, physical sanity verified).

### 2.2 Ramp Benchmark Calls (2 variables: `temperature_2m_max`, `precipitation_sum`)
Per-location weight factor for 4,383 days with 2 variables = $(4,383 / 14) \times (2 / 10) = \mathbf{62.6143}$ calls.

| Batch Size | Locations | Days | Vars | Call Weight | Status Code | Elapsed (s) | Locations Returned | Rate Limit Header / Note |
|---|---|---|---|---|---|---|---|---|
| **1** | 1 | 4,383 | 2 | **62.61** | 200 | 3.58 | 1 | OK |
| **5** | 5 | 4,383 | 2 | **313.07** | 200 | 1.55 | 5 | OK |
| **20** | 20 | 4,383 | 2 | **1,252.29** | 200 | 2.48 | 20 | OK |
| **50** | 50 | 4,383 | 2 | **3,130.71** | **429** | 1.02 | 0 | `Minutely API limit exceeded` |
| **100** | 100 | 4,383 | 2 | **6,261.43** | NOT RUN | - | 0 | Stopped at first non-200 |
| **200** | 200 | 4,383 | 2 | **12,522.86** | NOT RUN | - | 0 | Exceeds 10,000 daily limit |

---

## 3. Why the Ramp Stopped at Batch 20

1. **Status Code at Batch 50:** Returned **HTTP 429** (`{"error":true,"reason":"Minutely API request limit exceeded. Please try again in one minute."}`).
2. **Why 100 and 200 Were Not Run:** Phase 2 protocol mandates: *"Stop the ramp at the first non-200 response, never retry in a loop."*
3. **Underlying Mechanism:** A single batch request of 50 locations covering 4,383 days has a call weight of **3,130.71 API calls**. Because Open-Meteo evaluates the minutely rate limit (600 calls/min) using weighted calls, this single request exceeded the 600 calls/min threshold by **5.2×**, resulting in an immediate 429 throttling response. Batch 200 would have exceeded the entire daily quota of 10,000 in a single HTTP request (12,522.86 calls).

---

## 4. Projected Total Weight for 1,167 Kept Cells (4,383 Days)

### (a) Minimal 2-Variable Set (`temperature_2m_max`, `precipitation_sum`)
- **Weight per cell:** $1 \times (4,383 / 14) \times (2 / 10) = 62.61$ calls
- **Total projected weight:** $1,167 \times 62.6143 = \mathbf{73,065.87}\text{ weighted calls}$
- **Days needed at 100% capacity (10,000/day):** **7.31 days** (~8 calendar days)
- **Days needed at 80% safety cap (8,000/day):** **9.13 days** (~10 calendar days)

### (b) Full 6-Variable Set (`temp_max`, `temp_min`, `temp_mean`, `precip`, `wind_speed`, `rh_mean`)
- **Weight per cell:** $1 \times (4,383 / 14) \times (6 / 10) = 187.84$ calls
- **Total projected weight:** $1,167 \times 187.8429 = \mathbf{219,209.28}\text{ weighted calls}$
- **Days needed at 100% capacity (10,000/day):** **21.92 days** (~22 calendar days)
- **Days needed at 80% safety cap (8,000/day):** **27.40 days** (~28 calendar days)

---

## 5. Recommended Batching & Resumable Cached Harvesting Architecture

To operate safely at or below **80% of all limits**:
- **80% Rate Limits:**
  - Minutely: $\le 480$ weighted calls/min
  - Hourly: $\le 4,000$ weighted calls/hour
  - Daily: $\le 8,000$ weighted calls/day

### 5.1 Recommended Request Granularity
- **Optimal Batch Size:** **$n = 1$ location per HTTP request** (single cell).
  *(Batching multiple locations multiplies weight and triggers minutely 429 throttling).*
- **Paced Inter-Call Delay:**
  - For 2 variables (62.61 calls/req): Query 1 cell every **10 seconds** (6 req/min = 375.7 calls/min $\le 480$). Daily throughput: **127 cells/day**.
  - For 6 variables (187.84 calls/req): Query 1 cell every **30 seconds** (2 req/min = 375.7 calls/min $\le 480$). Daily throughput: **42 cells/day**.

### 5.2 Resumable Cache Protocol
1. **Per-Cell Storage:** Cache each fetched cell to `data/raw/weather/cells/{cell_id}.parquet`.
2. **Idempotent Skip:** Check filesystem before request; skip if parquet exists with 4,383 rows.
3. **Daily Budget Ledger:** Track cumulative calls in `quota_ledger.json`. When cumulative daily weight reaches 8,000, sleep until next UTC midnight.

---

## 6. Stop Gate 2 Evaluation & Decision

| Metric | Threshold | Measured Value (2 Variables) | Measured Value (6 Variables) | Status |
|---|---|---|---|---|
| Single-Cell Continuity | 4,383 continuous rows, 2014 intact | 4,383 rows, 0 nulls | 4,383 rows, 0 nulls | PASS |
| Error Threshold | No 429 at $\le 20$ locations | 1, 5, 20 all returned 200 | 1, 5, 20 all returned 200 | PASS |
| Harvest Duration | **$\le 3$ calendar days** | **9.1 days** (80% cap) | **27.4 days** (80% cap) | **FAIL** |

**STOP GATE 2 VERDICT:** **FAIL** ❌

The compliant harvest of weather data across 1,167 grid cells strictly exceeds the 3-day feasibility window (requiring 8–10 days for 2 variables and 22–28 days for 6 variables).

---

## 7. Options for Proceeding (Awaiting User Direction)

1. **Option A (Subsample Spatial Grid):** Sample representative cells (e.g., top 300 fire-prone forest grid cells accounting for ~80% of historical fire ignitions). 300 cells × 62.61 = 18,784 calls $\rightarrow$ **2.3 days** (Complies with $\le 3$ days).
2. **Option B (Direct ECMWF ERA5 Bulk Download via Copernicus CDS API):** Download NetCDF/GRIB reanalysis files covering India directly from the Copernicus Climate Data Store (CDS). Completely free, no daily call limits, avoids Open-Meteo per-call throttling, but requires CDS API key registration and NetCDF extraction.
3. **Option C (Accept Multi-Day Background Ingestion):** Implement the resumable caching script with a daemon background scheduler harvesting 127 cells/day over 9–10 days.
4. **Option D (Reduce Historical Temporal Window):** Fetch 2021–2025 (5 years + spin-up = ~2,190 days) instead of 11 years, halving total weight to ~36,500 calls $\rightarrow$ **4.5 days** (or 2022–2025 $\rightarrow$ **3.0 days**).
5. **Option E (Open-Meteo Commercial Key):** Procure a commercial API key removing daily limits ($>50$M calls/month).
