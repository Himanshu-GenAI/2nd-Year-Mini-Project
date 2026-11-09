# Week 2 Milestone Summary & Status Note

**Date:** 2026-11-09 | **Scope:** India Wildfire Early Alert (PS-113) — Week 2 Complete

---

## 1. What Was Built & Accomplished

During Week 2, the production data pipeline, rate-limited weather harvesting architecture, target label matrix, leakage-proof validation harness, and historical climatology benchmarks were constructed:
1. **Repository Clean & Prototypes Archived (Day 0):** Obsolete prototypes moved to `archive/prototype_v0/`; dependencies installed (`scikit-learn`, `matplotlib`, `pytest-cov`); Week 2 configuration and paths frozen in `src/config.py`.
2. **Weather Harvesting Pipeline (Day 1):** Built `src/data/quota.py` and `src/data/weather.py` implementing rolling 1m (480), 1h (4,000), and 24h (8,000) caps with atomic writes and mutual exclusion locking. Harvested and verified initial live cells.
3. **FIRMS Ingestion & 0.5° Grid Assignment (Day 2):** Built `src/data/grid.py` and `src/data/firms.py`. Filtered 923,584 raw detections into 811,280 clean observations within 1,167 grid cells, exactly reproducing the Week 1 audit.
4. **Fire Cell-Day Panel & Target Labels (Day 3):** Built `src/data/labels.py` creating the full balanced matrix of 4,684,338 cell-days across 4,014 dates. Generated `burning_prev3`, `fire_next_48h`, and `new_fire_next_48h` target labels.
5. **Leakage Harness & Evaluator (Day 4):** Built `src/utils/fit_registry.py` and `src/models/evaluate.py`. Implemented shift tests, split separation tests (2-day gap days strictly enforced), PR-AUC, Recall@10%, Precision@5%, and Brier score calculators.
6. **Climatology Benchmark (Day 5):** Built `src/models/dataset.py` (with locked test split) and `src/models/baselines.py`. Trained on 2015–2022 and scored on the 2023 validation set.
7. **KBDI Drought Engine (Day 6):** Built `src/features/kbdi.py` matching USDA SE-38 differential equations with 0.20-inch spell interception and 50°F floor. Generated demonstration trajectory `docs/figures/kbdi_demo.png`.

---

## 2. Benchmark Results on Validation Set (2023)

| Model | Variant | Set | Rows | Positives | Base Rate | PR-AUC | PR/Base Ratio | Recall@10% | Precision@5% | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| **Climatology** | `fire_next_48h` | All | 423,621 | 38,641 | 9.12% | **0.4272** | **4.68x** | 42.85% | 39.16% | 0.064422 |
| **Climatology** | `new_fire_next_48h` | Primary | 373,605 | 18,410 | 4.93% | **0.2198** | **4.46x** | 37.70% | 22.60% | 0.042091 |

---

## 3. Weather Harvesting Progress

- **Cached:** 3 cells (4,383 days/cell, 0 nulls) | **Remaining:** 1,164 cells
- **Harvest Schedule:** ~63 cells/hour (127 cells/day) over ~9.2 days (Target: Nov 18, 2026)
- **Quota Ledger:** 187.8 weight consumed in rolling 24h; 0 errors.

---

## 4. Traceability Table

| Metric / Value | Description | Output Source File |
|---|---|---|
| `923,584` | Raw FIRMS input rows | `docs/audit/firms_load_counts.csv` |
| `834,958` | Detections after filters (`type==0`, `conf>=30`) | `docs/audit/firms_load_counts.csv` |
| `811,280` | Clean detections in 1,167 kept cells | `docs/audit/firms_load_counts.csv` |
| `4,684,338` | Total cell-days in fire matrix | `docs/audit/label_stats.csv` |
| `3,405,306` | Training set cell-days (2015–2022) | `docs/audit/label_stats.csv` |
| `423,621` | Validation set cell-days (2023) | `docs/audit/label_stats.csv` |
| `850,743` | Test set cell-days (2024–2025) | `docs/audit/label_stats.csv` |
| `0.4272` | Climatology PR-AUC (`fire_next_48h`) | `docs/results.csv` |
| `0.2198` | Climatology PR-AUC (`new_fire_next_48h`) | `docs/results.csv` |

---

## 5. Week 3 Risk Flags & Inputs
1. **Severe Class Imbalance:** Primary new-fire incidence is 4.18% (train) and 4.93% (val).
2. **Weather Cache Ingestion:** Feature pipeline requires full 1,167 weather Parquets before full-matrix lag feature extraction.
