# NASA FIRMS Data Audit Note (MODIS C6.1 India 2015–2025)

**Audit Execution Date:** 2026-10-10  
**Source Dataset:** `data/raw/firms/fire_archive_M-C61_808434.csv` (MODIS C6.1 Historical Thermal Anomalies)  
**Sensors:** MODIS (Terra and Aqua satellites)

---

## 1. Dataset Overview and Synopsis Discrepancy

- **Total Raw Rows:** 923,584
- **Synopsis Claimed Count:** 1,880,000 (~1.88M)
- **Ratio to Synopsis Figure:** 0.4913 (49.13%)
- **Discrepancy Note:** The synopsis estimate of 1.88M was unverified. The actual raw dataset contains exactly **923,584** records covering 2015-01-01 to 2025-12-31 across India.
- **Instruments / Satellites:** Instrument is 100% `MODIS`. Satellites present: `Terra` (T) and `Aqua` (A). No non-MODIS or NRT data found.

---

## 2. Data Filtering and Attrition Waterfall

| Pipeline Stage | Rows Remaining | Retention vs. Prior Stage | Retention vs. Raw | Notes |
|---|---|---|---|---|
| **1. Raw Detections** | 923,584 | 100.0% | 100.0% | Input archive |
| **2. Deduplication** | 923,584 | 100.00% | 100.00% | Dropped 0 duplicate key records |
| **3. Confidence >= 30** | 884,467 | 95.76% | 95.76% | Filtered low-confidence noise |
| **4. Type == 0 (Presumed Vegetation)** | 834,958 | 94.40% | 90.40% | Excluded industrial/static thermal anomalies |

**Fire Type Column Verdict:** **`EXISTS`** (Type column is fully populated with integer codes 0, 2, 3; type 0 represents presumed vegetation fire).

---

## 3. Year-by-Year Detection Statistics

Median annual raw detections: **81,615.0**

| Year | Raw Detections | Kept Detections (`type==0`, `conf>=30`) | % of Median Raw | Flags |
|---|---|---|---|---|
| 2015 | 68,626 | 61,306 | 84.1% | Normal |
| 2016 | 88,987 | 81,053 | 109.0% | Normal |
| 2017 | 82,665 | 74,340 | 101.3% | Normal |
| 2018 | 91,197 | 82,040 | 111.7% | Normal |
| 2019 | 75,623 | 68,005 | 92.7% | Normal |
| 2020 | 76,113 | 68,641 | 93.3% | Normal |
| 2021 | 111,318 | 102,626 | 136.4% | Normal |
| 2022 | 81,615 | 74,275 | 100.0% | Normal |
| 2023 | 78,494 | 70,763 | 96.2% | Normal |
| 2024 | 74,029 | 66,246 | 90.7% | Normal |
| 2025 | 94,917 | 85,663 | 116.3% | Normal |

---

## 4. Monthly Distribution Profile (Pooled 2015–2025)

Wildfire activity in India is highly seasonal, peaking strongly during the pre-monsoon dry season (March–May) with a secondary post-monsoon agricultural burning peak (October–November).

| Month | Month Name | Raw Detections | Share (%) |
|---|---|---|---|
| 01 | January | 51,499 | 5.58% |
| 02 | February | 84,174 | 9.11% |
| 03 | March | 238,589 | 25.83% |
| 04 | April | 182,564 | 19.77% |
| 05 | May | 81,682 | 8.84% |
| 06 | June | 11,818 | 1.28% |
| 07 | July | 3,303 | 0.36% |
| 08 | August | 3,423 | 0.37% |
| 09 | September | 5,391 | 0.58% |
| 10 | October | 71,073 | 7.70% |
| 11 | November | 134,285 | 14.54% |
| 12 | December | 55,783 | 6.04% |

---

## 5. Issues Found & Anomaly Audit

- **Zero-Row Months:** None (all 132 calendar months from 2015-01 to 2025-12 have active fire records).
- **Out-of-Bounding-Box Records:** 0 records fall outside the broad bounding box (lat [6.0, 38.0], lon [67.0, 98.0]). These will be handled during spatial gridding.
- **NRT Records:** 0 (version is verified standard science archive C6.1).
- **Non-numeric Confidence:** 0 null or non-numeric confidence values.
- **Flagged Anomalies:**
  - Non-CSV file present: .gitkeep

---

## 6. Stop Gate 1 Evaluation

| Check | Requirement | Result | Status |
|---|---|---|---|
| File Parse | All CSV files parse successfully | 1/1 files parsed (0 errors) | **PASS** |
| `type` Column | Exists in all files | Present with 0/2/3 codes | **PASS** |
| Year Coverage | 2015–2025 all present with >0 rows | 11/11 years present | **PASS** |
| NRT Data | No NRT versions | 0 NRT records | **PASS** |
| Monthly Completeness | No zero-row months | 132/132 months active | **PASS** |
| Confidence Column | Numeric dtype, 0 nulls | int64, 0 nulls | **PASS** |
| Sensors / Satellite | MODIS (Terra/Aqua) only | MODIS Terra/Aqua only | **PASS** |

**STOP GATE 1 VERDICT:** **PASS** ✅
