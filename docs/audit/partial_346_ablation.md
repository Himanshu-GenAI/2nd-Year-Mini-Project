# Partial Run Audit: Features & Ablation (`partial: 346 cells`)

**Status:** PASS (no tuning, no Model v1 winner, no calibration run; `ALLOW_TEST_SPLIT` unset)  
**Evaluation Scope:** `val` (2023), primary label `new_fire_next_48h`, `set_name = "partial: 346 cells"`, `110,484` rows (`5,669` positives, base rate `0.051311`, `348` dates used, `eval_date = 2026-11-12`).  
**Sources:** `docs/audit/ablation.csv`, `docs/results.csv`, `docs/runs.csv`, `docs/data_audit_weather_complete.md`.

## 1. Ablation & Identical-Cell Baselines (`partial: 346 cells`)

Two-stage XGBoost training (Stage 1: fit 2015-2021, early-stop 2022; Stage 2: refit 2015-2022; score 2023 primary set), using default hyperparameters (`max_depth=6`, `learning_rate=0.1`).

| Group / Model | Features | Best Trees | PR-AUC | PR-AUC / Base | Recall@10% | Precision@5% | Brier | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `climatology` | 0 | 0 | 0.222830 | 4.343 | 0.3759 | 0.2399 | 0.043791 | 0.00 |
| `xgb_v0_plain` | 7 | 226 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 | 17.72 |
| `G1` (fire + cal + static) | 7 | 226 | 0.252007 | 4.911 | 0.4061 | 0.2546 | 0.042828 | 16.86 |
| `G2` (`G1` + 12 weather cols) | 19 | 111 | 0.273351 | 5.327 | 0.4060 | 0.2602 | 0.042202 | 11.63 |
| `G3` (`G2` + 2 KBDI cols) | 21 | 113 | 0.271228 | 5.286 | 0.4048 | 0.2544 | 0.042249 | 13.04 |

### Incremental Changes
- **`G1 -> G2` (+12 weather features):** PR-AUC `+0.021344`, PR-AUC/base `+0.416`, Recall@10% `-0.0001`, Precision@5% `+0.0056`, Brier `-0.000626`.
- **`G2 -> G3` (+2 KBDI features, untuned):** PR-AUC `-0.002123`, PR-AUC/base `-0.041`, Recall@10% `-0.0012`, Precision@5% `-0.0058`, Brier `+0.000047`.
- **`G3` vs `climatology`:** PR-AUC `+0.048398` (`0.271228` vs `0.222830`).

## 2. State Sampling Note (`346` of `1,167` Cells)

- **State Coverage:** `30` of `30` grid states covered (`0` missing states).
- **States with `< 3` sampled cells (6 states):**
  - `Goa`: `1 / 1`
  - `Meghalaya`: `1 / 10`
  - `Mizoram`: `1 / 8`
  - `Nagaland`: `1 / 7`
  - `Tripura`: `1 / 4`
  - `Sikkim`: `2 / 4`

## 3. Verification & Stop Gates

- **Weather Cache Validation (`scripts/validate_weather_cache.py`):** `complete=346`, `missing=821`, `failed=0` (`total=1167`), exit code `0`.
- **Partial Feature Table (`data/processed/features_partial.parquet`):** `1,388,844` rows (`346 * 4,014`), `30` columns, `0` NaNs, `has_weather` all `True`; full `data/processed/features.parquet` absent.
- **Leakage & Unit Tests:** `18 passed` (`tests/test_leakage.py`, `tests/test_build_features.py`, `tests/test_weather_features.py`).
- **Cell List Invariance:** `data/processed/partial_cells.json` unchanged post-ablation (`346` cells).
- **Test Split Lock:** `ALLOW_TEST_SPLIT` unset; 2024-2025 test split unread.

