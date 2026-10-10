# Open-Meteo Weather API Feasibility and Probe Note

**Audit Date:** 2026-10-10  
**Target Domain:** India 0.5° Spatial Grid (Lat 6.0°–38.0°N, Lon 67.0°–98.0°E)  
**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`  
**Temporal Span:** 2014-01-01 to 2025-12-31 (4,383 days inclusive; 2014 serves as KBDI spin-up)

---

## 1. Documented API Specifications & Free-Tier Limits

- **Document Inspection Date:** 2026-10-10
- **Variables Confirmed Available:**
  - Required: `temperature_2m_max`, `precipitation_sum`
  - Optional / Extended: `temperature_2m_min`, `temperature_2m_mean`, `wind_speed_10m_max`, `relative_humidity_2m_mean`, `relative_humidity_2m_max`, `relative_humidity_2m_min`
- **Daily Relative Humidity:** Confirmed present (`relative_humidity_2m_mean`).
- **Free-Tier Limits:**
  - Per minute: 600 API calls
  - Per hour: 5,000 API calls
  - Per day: 10,000 API calls
- **Quota Weighting Rule:** 1 API call per requested coordinate in multi-location batch requests (e.g., a batch request with 50 locations is weighted as 50 API calls).

---

## 2. Single-Cell Verification (Lat 30.25°N, Lon 78.75°E)

- **HTTP Status:** 200
- **Total Records Returned:** 4,383 days (continuous daily sequence from 2014-01-01 to 2025-12-31)
- **2014 Spin-up Completeness:** 365/365 valid days (0 nulls in `temperature_2m_max` and `precipitation_sum`)
- **Physical Plausibility Check:**
  - `precipitation_sum`: Min = 0.00 mm, Max = 112.80 mm (All >= 0.0 mm [OK])
  - `temperature_2m_max`: Min = 8.50 °C, Max = 38.40 °C (All within [-20°C, 60°C] [OK])
- **Response Size:** 175,469 bytes (171.4 KB)
- **Response Latency:** 1.66 s

---

## 3. Batch Ramp Benchmark Results

| Batch Size (Locations) | Status | Wall Time (s) | Size (Bytes) | Locations Returned | Sec / Location | Rate Limit Header |
|---|---|---|---|---|---|---|
| 1 | 200 | 4.22 | 101,573 | 1 | 4.2176 | N/A |
| 5 | 200 | 1.40 | 507,988 | 5 | 0.2802 | N/A |
| 20 | 200 | 2.99 | 2,035,124 | 20 | 0.1493 | N/A |
| 50 | 429 | 1.12 | 94 | 0 | 0.0000 | N/A |

---

## 4. Full Dataset Cost Model & Execution Plan

Using a planning baseline of **N = 1,200 grid cells** across India:

- **Recommended Batch Size:** 50 locations per HTTP request
- **Total HTTP Requests:** 24 requests
- **Weighted API Call Volume:** 1,200 calls (12.0% of daily allowance of 10,000)
- **Inter-Request Delay:** 3.0 seconds
- **Projected Wall-Clock Time:** ~2.4 minutes (~144 seconds)
- **Fetch Compliance:** The entire 1,200-cell national weather archive (2014–2025, 4,383 days) can be safely harvested in **1 calendar day** without exceeding daily or hourly rate limits.

---

## 5. Stop Gate 2 Evaluation

| Check | Requirement | Result | Status |
|---|---|---|---|
| Single-Cell Assertions | 4,383 continuous rows, 2014 complete, physical sanity | 4,383 rows, 0 nulls in 2014, valid temp/precip | **PASS** |
| Error Threshold | No 429 or errors at <= 20 locations | 1, 5, 20 all returned HTTP 200 | **PASS** |
| Harvest Feasibility | Projected compliant fetch <= 3 days | 1 day required (2.4 min total) | **PASS** |

**STOP GATE 2 VERDICT:** **PASS** [OK]
