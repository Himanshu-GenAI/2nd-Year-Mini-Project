"""End-to-end smoke test for Week 2 integration."""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.firms import load_clean_detections  # noqa: E402
from src.data.grid import load_grid  # noqa: E402
from src.models.baselines import fit_climatology, predict_climatology  # noqa: E402
from src.models.dataset import load_split  # noqa: E402
from src.models.evaluate import evaluate  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


def get_mem_mb() -> float:
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def run_smoke_test():
    print("=" * 60)
    print("END-TO-END INTEGRATION SMOKE TEST (WEEK 2)")
    print("=" * 60)

    # Step 1: Load Grid
    t0 = time.perf_counter()
    m0 = get_mem_mb()
    grid_df = load_grid()
    t1 = time.perf_counter()
    m1 = get_mem_mb()
    print(f"1. load_grid(): {len(grid_df)} cells | Time: {t1 - t0:.3f}s | Mem: {m1:.1f} MB (diff: {m1 - m0:+.1f} MB)")

    # Step 2: Load Clean Detections
    t0 = time.perf_counter()
    m0 = get_mem_mb()
    clean_df = load_clean_detections()
    t1 = time.perf_counter()
    m1 = get_mem_mb()
    print(f"2. load_clean_detections(): {len(clean_df):,} detections | Time: {t1 - t0:.3f}s | Mem: {m1:.1f} MB (diff: {m1 - m0:+.1f} MB)")

    # Step 3: Load Fire Table
    t0 = time.perf_counter()
    m0 = get_mem_mb()
    fire_table = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "fire_cell_day.parquet")
    t1 = time.perf_counter()
    m1 = get_mem_mb()
    print(f"3. load_fire_table(): {len(fire_table):,} cell-days | Time: {t1 - t0:.3f}s | Mem: {m1:.1f} MB (diff: {m1 - m0:+.1f} MB)")

    # Step 4: Load Validation Split (without touching test split)
    t0 = time.perf_counter()
    m0 = get_mem_mb()
    val_prim = load_split("val", "new_fire_next_48h", primary=True)
    t1 = time.perf_counter()
    m1 = get_mem_mb()
    print(f"4. load_split('val', primary=True): {len(val_prim):,} rows | Time: {t1 - t0:.3f}s | Mem: {m1:.1f} MB (diff: {m1 - m0:+.1f} MB)")

    # Step 5: Fit Climatology on Train & Score Validation
    t0 = time.perf_counter()
    m0 = get_mem_mb()
    train_prim = load_split("train", "new_fire_next_48h", primary=True)
    model = fit_climatology(train_prim, "new_fire_next_48h", alpha=1.0)
    val_prim["score"] = predict_climatology(model, val_prim)
    t1 = time.perf_counter()
    m1 = get_mem_mb()
    print(f"5. fit & predict_climatology(): {len(val_prim):,} scored | Time: {t1 - t0:.3f}s | Mem: {m1:.1f} MB (diff: {m1 - m0:+.1f} MB)")

    # Step 6: Evaluate
    t0 = time.perf_counter()
    m0 = get_mem_mb()
    res = evaluate(val_prim, "new_fire_next_48h", "score", "climatology", "val", "primary")
    t1 = time.perf_counter()
    m1 = get_mem_mb()
    print(f"6. evaluate(): PR-AUC={res['pr_auc']:.4f} (Base={res['base_rate']:.4f}) | Time: {t1 - t0:.3f}s | Mem: {m1:.1f} MB (diff: {m1 - m0:+.1f} MB)")

    print("\nALL END-TO-END SMOKE TEST STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_smoke_test()
