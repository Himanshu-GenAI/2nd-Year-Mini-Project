"""Run Day 5 Climatology Baseline Evaluation on 2023 Validation Split."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import fit_climatology, predict_climatology  # noqa: E402
from src.models.dataset import load_split  # noqa: E402
from src.models.evaluate import (  # noqa: E402
    evaluate,
    plot_reliability,
    record_result,
    reliability_table,
)


def run_baselines():
    print("=" * 60)
    print("DAY 5: CLIMATOLOGY BASELINE EVALUATION (VAL 2023)")
    print("=" * 60)

    # 1. Variant A: fire_next_48h (All cell-days)
    print("\n--- 1. Evaluating Variant A: fire_next_48h (All rows) ---")
    train_df_all = load_split("train", label="fire_next_48h", primary=False)
    val_df_all = load_split("val", label="fire_next_48h", primary=False)

    print(f"Train rows: {len(train_df_all):,}, Val rows: {len(val_df_all):,}")
    model_any = fit_climatology(train_df_all, label="fire_next_48h", alpha=1.0)
    val_df_all["score_climatology"] = predict_climatology(model_any, val_df_all)

    res_any = evaluate(
        df=val_df_all,
        label="fire_next_48h",
        score="score_climatology",
        model="climatology",
        split="val",
        set_name="all",
    )
    record_result(res_any)

    # Plot reliability
    rel_table_any = reliability_table(val_df_all, label="fire_next_48h", score="score_climatology")
    plot_path_any = PROJECT_ROOT / "docs" / "figures" / "reliability_climatology_fire_next_48h.png"
    plot_reliability(rel_table_any, plot_path_any)
    print(f"Reliability plot saved to {plot_path_any}")

    print("Variant A Results:")
    for k, v in res_any.items():
        print(f"  {k}: {v}")

    # 2. Variant B: new_fire_next_48h (Primary set: burning_prev3 == False)
    print("\n--- 2. Evaluating Variant B: new_fire_next_48h (Primary set) ---")
    train_df_prim = load_split("train", label="new_fire_next_48h", primary=True)
    val_df_prim = load_split("val", label="new_fire_next_48h", primary=True)

    print(f"Train rows (primary): {len(train_df_prim):,}, Val rows (primary): {len(val_df_prim):,}")
    model_new = fit_climatology(train_df_prim, label="new_fire_next_48h", alpha=1.0)
    val_df_prim["score_climatology"] = predict_climatology(model_new, val_df_prim)

    res_new = evaluate(
        df=val_df_prim,
        label="new_fire_next_48h",
        score="score_climatology",
        model="climatology",
        split="val",
        set_name="primary",
    )
    record_result(res_new)

    # Plot reliability
    rel_table_new = reliability_table(val_df_prim, label="new_fire_next_48h", score="score_climatology")
    plot_path_new = PROJECT_ROOT / "docs" / "figures" / "reliability_climatology_new_fire_next_48h.png"
    plot_reliability(rel_table_new, plot_path_new)
    print(f"Reliability plot saved to {plot_path_new}")

    print("Variant B Results:")
    for k, v in res_new.items():
        print(f"  {k}: {v}")

    # Verify Stop Gate: PR-AUC on val must be above base rate
    assert res_any["pr_auc"] > res_any["base_rate"], "Stop Gate Fail: fire_next_48h PR-AUC not above base rate"
    assert res_new["pr_auc"] > res_new["base_rate"], "Stop Gate Fail: new_fire_next_48h PR-AUC not above base rate"
    print("\nSTOP GATE PASSED: PR-AUC on val is strictly above base rate for both variants!")


if __name__ == "__main__":
    run_baselines()
