# Decision Log

| ID | Date | Decision | Rationale | Evidence File | Owner | Status |
|---|---|---|---|---|---|---|
| D-GRID | 2026-10-10 | Spatial grid resolution set to 0.5 degrees spanning lat [6.0, 38.0], lon [67.0, 98.0] (64x62 = 3,968 candidate cells). | Balances spatial granularity with compute feasibility for multi-year national weather and fire modeling across India. | `notebooks/03_grid_probe.py` | Team | provisional |
| D-SPLIT | 2026-10-10 | Train: 2015-01-03 to 2022-12-29; Val: 2023-01-01 to 2023-12-29; Test: 2024-01-01 to 2025-12-29; with 2-day buffer gaps (Dec 30-31) between splits. | Prevents temporal data leakage given the 2-day forward label horizon and 3-day lookback window. | `src/config.py` | Team | provisional |
