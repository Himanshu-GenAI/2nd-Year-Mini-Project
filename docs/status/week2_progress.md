# Week 2 Harvesting & Modeling Progress Report

**Date:** 2026-11-09 (Monday, Day 1-5 Track)  
**Project:** Wildfire Early Alert System for India (PS-113)

---

## 1. Weather Harvester Progress

- **Total Grid Cells:** 1,167
- **Cached Cells (4,383 days):** 3 (0.3%)
- **Failed Cells:** 0
- **Pacing & Caps:** 480/min, 4,000/hr, 8,000/day; paced at ~63 cells/hour (1 request every ~55–60s)
- **Daily Target Rate:** ~127 cells/day
- **Projected Completion ETA:** ~9.2 days (target completion: 18 Nov 2026)
- **Quota Ledger State:** 187.8 weight consumed in 24h rolling window; zero 429 rate limit errors.

---

## 2. Climatology Baseline Benchmark Results (2023 Validation Split)

| Variant | Split | Eval Set | Evaluated Rows | Positives | Base Rate | PR-AUC | PR/Base Uplift | Recall@10% | Precision@5% | Brier Score |
|---|---|---|---|---|---|---|---|---|---|---|
| **`fire_next_48h`** | Val (2023) | All | 423,621 | 38,641 | 9.12% | **0.4272** | **4.68x** | 42.85% | 39.16% | 0.064422 |
| **`new_fire_next_48h`** | Val (2023) | Primary | 373,605 | 18,410 | 4.93% | **0.2198** | **4.46x** | 37.70% | 22.60% | 0.042091 |

---

## 3. Milestones & Gates Passed

- [x] **D5 & D6 Compliant:** Zero temporal leakage, 2-day gap days respected, rate limits strictly observed.
- [x] **Stop Gate Check:** Climatology baseline PR-AUC strictly beats empirical base rate for both variants (4.68x and 4.46x baseline ratio).
- [x] **Test Split Isolation:** Test split (2024–2025) locked and untouched.
