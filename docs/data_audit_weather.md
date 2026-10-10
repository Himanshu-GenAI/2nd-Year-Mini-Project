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

1. **Status Code at Batch 50:** Returned **HTTP 429** (`{"reason":"Minutely API request limit exceeded. Please try again in one minute.","error":true}`).
2. **Why 100 and 200 Were Not Run:** Phase 2 protocol mandates: *"Stop the ramp at the first non-200 response, never retry in a loop."*
3. **Corrected Assessment of 429 Mechanism:** The 20-location call (weight 1,252.29 calls) returned HTTP 200 despite exceeding 600 weight. Therefore, an individual request exceeding 600 weight does not automatically trigger an immediate block by itself. Rather, the likely cause of the HTTP 429 at batch 50 is **accumulated weight within the rolling minute** across consecutive probe requests (Batch 1 + Batch 5 + Batch 20 + Batch 50), stated here as an untested hypothesis. Per protocol, no additional probes were run.

---

## 4. Projected Total Weight for 1,167 Kept Cells (4,383 Days)

### (a) Decided 2-Variable Set (`temperature_2m_max`, `precipitation_sum`)
- **Weight per cell:** $1 \times (4,383 / 14) \times (2 / 10) = 62.61$ calls
- **Total projected weight:** $1,167 \times 62.6143 = \mathbf{73,065.87}\text{ weighted calls}$
- **Days needed at 100% capacity (10,000/day):** **7.31 days** (~8 calendar days)
- **Days needed at 80% safety cap (8,000/day):** **9.13 days** (~10 calendar days)

### (b) Extended 6-Variable Set (`temp_max`, `min`, `mean`, `precip`, `wind`, `rh`)
- **Weight per cell:** $1 \times (4,383 / 14) \times (6 / 10) = 187.84$ calls
- **Total projected weight:** $1,167 \times 187.8429 = \mathbf{219,209.28}\text{ weighted calls}$
- **Days needed at 100% capacity (10,000/day):** **21.92 days** (~22 calendar days)
- **Days needed at 80% safety cap (8,000/day):** **27.40 days** (~28 calendar days)
- *Decision note:* Postponed per D6 to avoid 28-day ingestion delay.

---

## 5. Decided Harvesting Architecture & Schedule (Option C, ≤80% Limits)

To operate strictly within **80% of all rate limits**:
- **80% Rate Limit Caps:**
  - Minutely: $\le 480$ weighted calls/min
  - Hourly: $\le 4,000$ weighted calls/hour
  - Daily: $\le 8,000$ weighted calls/day (daily counter verified to reset at 00:00 UTC)

### 5.1 Recomputed Compliant Schedule (Hourly & Daily Caps)
- **Variables:** `temperature_2m_max` + `precipitation_sum` only (62.61 calls weight per cell).
- **Batch Granularity:** $n = 1$ cell per HTTP request.
- **Hourly Cap Throughput:**
  $$\frac{4,000\text{ weight/hour}}{62.6143\text{ weight/cell}} \approx 63.88 \rightarrow \mathbf{\approx 63\text{ cells / hour}}$$
  *(63 cells/hour $\times 62.6143 = 3,944.7$ weight/hour $\le 4,000$ cap).*
- **Pacing:** Query 1 cell every **57–60 seconds** (average 1 call/min = 62.61 calls/min, well below 480 calls/min cap).
- **Daily Cap Throughput:**
  $$\frac{8,000\text{ weight/day}}{62.6143\text{ weight/cell}} \approx 127.77 \rightarrow \mathbf{127\text{ cells / day}}$$
  *(127 cells/day $\times 62.6143 = 7,952.0$ weight/day $\le 8,000$ cap; represents ~2 hours of active hourly ingestion per day).*
- **Total Duration:**
  $$\frac{1,167\text{ cells}}{127\text{ cells/day}} \approx 9.19\text{ days} \rightarrow \mathbf{\approx 9\text{--}10\text{ calendar days}}.$$

### 5.2 Resumable Cache Protocol
1. **Per-Cell Parquet Cache:** Store fetched cells to `data/raw/weather/cells/{cell_id}.parquet`.
2. **Idempotent Skip:** Check filesystem before request; skip if parquet exists with 4,383 rows.
3. **Rolling Quota Ledger:** Maintain `quota_ledger.json` tracking timestamps and weight across sliding 1-min, 1-hr, and 24-hr windows. Pauses automatically if hourly cap (4,000) or daily cap (8,000) is reached.

---

## 6. Stop Gate 2 Evaluation & Decision

| Metric | Threshold | Measured Value (2 Variables) | Status |
|---|---|---|---|
| Single-Cell Continuity | 4,383 continuous rows, 2014 intact | 4,383 rows, 0 nulls | PASS |
| Error Threshold | No 429 at $\le 20$ locations | 1, 5, 20 returned 200 | PASS |
| Harvest Duration | $\le 3$ calendar days | **9.1 days** (80% cap) | **FAIL, resolved by D6** |

**STOP GATE 2 VERDICT:** **FAIL, resolved by D6** ✅

### Resolution via Decision D6:
1. **Chosen Option C:** Accept multi-day background ingestion (~10 days, 127 cells/day).
2. **Rejected Alternatives:**
   - **Rejected Option A:** Subsampling based on 2015–2025 fire counts leaks test-period label information into spatial selection and artificially inflates PR-AUC.
   - **Rejected Option D:** Reducing the temporal window sacrifices 11-year climate baseline and KBDI spin-up.
   - **Rejected Option E:** Commercial API key rejected for open student reproduction.
3. **Immediate Action:** Do not start weather fetch yet. Features using humidity or wind are postponed; direct Copernicus CDS ERA5 bulk ingestion will be evaluated in Week 2.
