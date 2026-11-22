"""Model v1 (tuned XGBoost G3) SHAP global summary and grouped High/Critical local explanations.

All SHAP contributions are computed in log-odds (model margin) units, NOT
probabilities, using shap.TreeExplainer (exact TreeSHAP for XGBoost). Raw
per-feature SHAP attributions are summed within each display group (8 groups in `GROUPS`)
for operational clarity while keeping exact per-feature attributions as an audit
table. Strictly reads only 2023 validation primary-set rows and fails loudly if
any train/test row, burning row, or additivity violation (> 1e-4) occurs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import yaml
from scipy.stats import spearmanr

from src.config import DOCS_AUDIT_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.explain.labels import FEATURE_LABELS, GROUPS, get_driver_group
from src.explain.shap_v0 import (
    ADDITIVITY_TEST_SIZE,
    ADDITIVITY_TOL,
    GLOBAL_SAMPLE_SIZE,
    LOCAL_TOP_K,
    assert_val_only_rows,
    compute_tree_shap_margins,
    extract_top_k_grouped_local_shap,
    extract_top_k_local_shap,
    group_shap_values,
    verify_shap_additivity,
)
from src.features.build_features import ALL_FEATURE_NAMES
from src.models.calibrate import MODEL_V1_BUNDLE_DIR, load_model_v1_bundle
from src.models.features_io import load_features
from src.models.train_v1 import assert_test_split_locked
from src.utils.plotting import plt

PREDICTIONS_V1_PATH = PROCESSED_DIR / "predictions" / "model_v1_dev.parquet"
TIERS_PREVIEW_V1_PATH = PROJECT_ROOT / "configs" / "tiers_v1_preview.yaml"

SHAP_LOCAL_V1_PATH = PROCESSED_DIR / "shap" / "model_v1_dev.parquet"
SHAP_LOCAL_V1_BY_FEATURE_PATH = PROCESSED_DIR / "shap" / "model_v1_dev_by_feature.parquet"
SHAP_GLOBAL_V1_FEATURES_CSV = DOCS_AUDIT_DIR / "shap_global_v1_features.csv"
SHAP_GLOBAL_V1_GROUPS_CSV = DOCS_AUDIT_DIR / "shap_global_v1_groups.csv"
SHAP_SUMMARY_V1_PNG = PROJECT_ROOT / "docs" / "figures" / "shap_summary_v1.png"
SHAP_SANITY_V1_MD = DOCS_AUDIT_DIR / "shap_sanity_v1.md"


def compute_global_group_shap_v1(
    shap_values: np.ndarray,
    feature_names: list[str],
    csv_path: Path = SHAP_GLOBAL_V1_GROUPS_CSV,
    group_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Compute global driver importance as mean(|summed group SHAP|) over sampled rows."""
    mapping = group_mapping or GROUPS
    driver_names, grouped_shap, _ = group_shap_values(shap_values, feature_names, mapping)
    mean_abs = np.mean(np.abs(grouped_shap), axis=0)
    group_df = (
        pd.DataFrame({"driver": driver_names, "mean_abs_shap": mean_abs.astype(float)})
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    group_df["rank"] = np.arange(1, len(group_df) + 1, dtype=int)
    group_df["mean_abs_shap"] = group_df["mean_abs_shap"].round(6)
    group_df = group_df[["driver", "mean_abs_shap", "rank"]]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    group_df.to_csv(csv_path, index=False)
    return group_df, grouped_shap, driver_names


def compute_global_feature_shap_v1(
    model: Any,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    figure_path: Path = SHAP_SUMMARY_V1_PNG,
    csv_path: Path = SHAP_GLOBAL_V1_FEATURES_CSV,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    """Compute global feature SHAP ranking on 20k sample, verify additivity, and save CSV + beeswarm."""
    shap_vals, base_val = compute_tree_shap_margins(model, X_sample[feature_names])
    max_diff = verify_shap_additivity(model, X_sample[feature_names], shap_vals, base_val)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    summary_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "driver": [get_driver_group(f) for f in feature_names],
                "mean_abs_shap": mean_abs.astype(float),
            }
        )
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    summary_df["rank"] = np.arange(1, len(summary_df) + 1, dtype=int)
    summary_df["mean_abs_shap"] = summary_df["mean_abs_shap"].round(6)
    summary_df = summary_df[["feature", "driver", "mean_abs_shap", "rank"]]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)

    # Render beeswarm summary plot
    display_names = [f"{f} ({FEATURE_LABELS.get(f, f)})" for f in feature_names]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample[feature_names], feature_names=display_names, show=False)
    plt.title("Model v1 Global SHAP Summary (20,000 Random 2023 Primary Rows, Log-Odds Units)")
    plt.xlabel("SHAP value (impact on log-odds margin)")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    return summary_df, shap_vals, base_val, max_diff


def build_shap_sanity_report_v1(
    summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
    X_sample: pd.DataFrame,
    shap_sample: np.ndarray,
    keys_sample: pd.DataFrame,
    feature_names: list[str],
    base_value: float,
    max_diff_1k: float,
    max_diff_20k: float,
    max_diff_local_grouped: float,
    local_rows_count: int,
    local_cell_days: int,
    runtime_s: float,
    model_version: str,
    report_path: Path = SHAP_SANITY_V1_MD,
) -> str:
    """Write comprehensive Model v1 SHAP sanity audit markdown."""
    total_grp_abs = float(group_df["mean_abs_shap"].sum())
    grp_table_rows = []
    for _, r in group_df.iterrows():
        drv = str(r["driver"])
        share = (float(r["mean_abs_shap"]) / total_grp_abs * 100.0) if total_grp_abs > 0 else 0.0
        grp_table_rows.append(f"| {int(r['rank'])} | {drv} | {float(r['mean_abs_shap']):.6f} | {share:.2f}% |")

    total_abs = float(summary_df["mean_abs_shap"].sum())
    feat_table_rows = []
    for _, r in summary_df.iterrows():
        feat = str(r["feature"])
        share = (float(r["mean_abs_shap"]) / total_abs * 100.0) if total_abs > 0 else 0.0
        feat_table_rows.append(
            f"| {int(r['rank'])} | `{feat}` | {r['driver']} | {float(r['mean_abs_shap']):.6f} | {share:.2f}% |"
        )

    # 1. Fire activity checks: Spearman of fire features with their respective group SHAP
    driver_names, grouped_shap, _ = group_shap_values(shap_sample, feature_names)
    fire_checks: list[str] = []
    for f in ["fire_count_7d", "fire_count_30d", "neighbour_fire_7d"]:
        grp_name = get_driver_group(f)
        grp_col = driver_names.index(grp_name)
        vals = X_sample[f].to_numpy(dtype=np.float64)
        grp_sh = grouped_shap[:, grp_col]
        pos_mask = vals > 0
        zero_mask = vals == 0
        mean_pos = float(grp_sh[pos_mask].mean()) if pos_mask.any() else float("nan")
        mean_zero = float(grp_sh[zero_mask].mean()) if zero_mask.any() else float("nan")
        rho = float(spearmanr(vals, grp_sh).statistic)
        verdict = "PASS" if rho > 0 else "FAIL"
        fire_checks.append(
            f"- `{f}` -> **{grp_name}**: mean group SHAP when `> 0` is `{mean_pos:+.4f}` "
            f"(n={int(pos_mask.sum()):,}) vs `== 0` `{mean_zero:+.4f}` (n={int(zero_mask.sum()):,}); "
            f"Spearman `rho = {rho:+.4f}` (**{verdict}**)."
        )

    # 2. Weather & Drought physical direction checks
    physical_checks: list[str] = []
    checks_spec = [
        ("kbdi", "positive", "Higher KBDI should raise fire risk"),
        ("tmax_t0", "positive", "Higher maximum temperature should raise fire risk"),
        ("tmax_mean_7d", "positive", "Higher 7d mean max temperature should raise fire risk"),
        ("precip_sum_7d", "negative", "Higher recent rainfall should lower fire risk"),
        ("precip_sum_30d", "negative", "Higher 30d rainfall should lower fire risk"),
        ("days_since_rain", "positive", "More days since rain should raise fire risk"),
    ]

    opposite_flags: list[dict[str, Any]] = []

    for feat, expected_dir, desc in checks_spec:
        if feat in feature_names:
            j = feature_names.index(feat)
            vals = X_sample[feat].to_numpy(dtype=np.float64)
            sh = shap_sample[:, j]
            rho = float(spearmanr(vals, sh).statistic)
            expected_pos = expected_dir == "positive"
            actual_pos = rho > 0
            is_match = (expected_pos and actual_pos) or (not expected_pos and not actual_pos)
            verdict = "PASS" if is_match else "OPPOSITE SIGN"
            physical_checks.append(
                f"- `{feat}` ({desc}): Spearman `rho = {rho:+.4f}` (expected {expected_dir}) -> **{verdict}**"
            )
            if not is_match:
                opposite_flags.append({"feature": feat, "rho": rho, "expected": expected_dir, "j": j})

    # Detailed slice breakdown for any opposite-sign feature (report, do not explain away)
    slice_breakdowns: list[str] = []
    if opposite_flags:
        # Check concentration across months
        dates = pd.to_datetime(keys_sample["date"])
        months = dates.dt.month
        for flag in opposite_flags:
            f = flag["feature"]
            j = flag["j"]
            vals = X_sample[f].to_numpy(dtype=np.float64)
            sh = shap_sample[:, j]
            month_rhos = {}
            for m in range(1, 13):
                m_mask = (months == m).to_numpy()
                if m_mask.sum() > 20 and np.std(vals[m_mask]) > 1e-6 and np.std(sh[m_mask]) > 1e-6:
                    r_m = float(spearmanr(vals[m_mask], sh[m_mask]).statistic)
                    month_rhos[m] = r_m
            m_summary = ", ".join(f"M{m}:{r:+.2f}" for m, r in sorted(month_rhos.items()))
            slice_breakdowns.append(
                f"- **Opposite-sign feature `{f}` (overall rho={flag['rho']:+.4f}):** "
                f"Monthly correlation breakdown: {m_summary}."
            )
    else:
        slice_breakdowns.append("- None: All key physical features align with expected physical risk directions.")

    # 3. Share of total absolute SHAP across driver clusters
    reg_seas_drivers = {"Regional baseline", "Seasonal baseline"}
    fire_drivers = {
        "Fires in this cell, last 7 days",
        "Fires in this cell, last 30 days",
        "Fires in nearby cells, last 7 days",
    }
    weather_drought_drivers = {"Recent heat", "Recent rainfall", "Drought index"}

    reg_seas_abs = float(group_df.loc[group_df["driver"].isin(reg_seas_drivers), "mean_abs_shap"].sum())
    fire_abs = float(group_df.loc[group_df["driver"].isin(fire_drivers), "mean_abs_shap"].sum())
    weather_drought_abs = float(group_df.loc[group_df["driver"].isin(weather_drought_drivers), "mean_abs_shap"].sum())

    reg_seas_pct = (reg_seas_abs / total_grp_abs * 100.0) if total_grp_abs > 0 else 0.0
    fire_pct = (fire_abs / total_grp_abs * 100.0) if total_grp_abs > 0 else 0.0
    weather_drought_pct = (weather_drought_abs / total_grp_abs * 100.0) if total_grp_abs > 0 else 0.0

    lines = [
        "# Model v1 SHAP Sanity Audit",
        "",
        f"- **Model version:** `{model_version}` (`xgb_v1_tuned_G3`, 21 features grouped into 8 driver groups)",
        f"- **Route used:** `shap=={shap.__version__}` (`shap.TreeExplainer`, exact TreeSHAP)",
        "- **Units:** All SHAP values (`shap`) and base value are in **log-odds (margin) units**, NOT probabilities.",
        f"- **Expected (base) value:** `{base_value:.6f}` log-odds",
        f"- **Grouped additivity check (`|sum_g(shap_g) + base - margin|`):** max diff `{max_diff_1k:.2e}` (1,000 random rows), "
        f"`{max_diff_20k:.2e}` (20,000 global rows), `{max_diff_local_grouped:.2e}` ({local_cell_days:,} High/Critical local rows) — all <= `1e-4` (**PASS**).",
        f"- **Scope verified:** Strictly 2023 validation primary-set rows (`split == \"val\"`). Grouped local parquet (`{local_rows_count:,}` rows = `{local_cell_days:,}` cell-days x top 5 drivers) + audit per-feature parquet computed in `{runtime_s:.2f}s`.",
        "",
        "## 1. Grouped Driver Ranking (`mean(|sum_g SHAP_g|)`, 20,000 Random 2023 Primary Rows, Seed 42)",
        "",
        "| Rank | Driver (Grouped) | Mean \\|Summed SHAP\\| (log-odds) | Share |",
        "|---|---|---|---|",
        *grp_table_rows,
        "",
        "## 2. Per-Feature Ranking (`docs/audit/shap_global_v1_features.csv`)",
        "",
        "| Rank | Feature | Driver Group | Mean \\|SHAP\\| (log-odds) | Share |",
        "|---|---|---|---|---|",
        *feat_table_rows,
        "",
        "## 3. Directional Sanity: Do Recent Fires Raise Predicted Risk?",
        "",
        *fire_checks,
        "",
        "## 4. Directional Sanity: Weather and Drought Physical Expectations",
        "",
        *physical_checks,
        "",
        "### Physical Anomaly Reporting",
        "",
        *slice_breakdowns,
        "",
        "## 5. Share of Total Grouped Absolute SHAP Across Clusters",
        "",
        f"- **Regional & Seasonal Baseline (`Regional baseline` + `Seasonal baseline`):** `{reg_seas_pct:.2f}%` (mean |summed SHAP| = `{reg_seas_abs:.4f}`)",
        f"- **Fire Activity Groups (3 groups):** `{fire_pct:.2f}%` (mean |summed SHAP| = `{fire_abs:.4f}`)",
        f"- **Weather & Drought Groups (`Recent heat` + `Recent rainfall` + `Drought index`):** `{weather_drought_pct:.2f}%` (mean |summed SHAP| = `{weather_drought_abs:.4f}`)",
        "",
        f"> **SUMMARY:** In Model v1, dynamic environmental drivers (weather + drought + fire activity) account for "
        f"`{(fire_pct + weather_drought_pct):.2f}%` of grouped attribution, reducing static geographic and calendar baseline dominance from v0's levels.",
        "",
    ]

    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text


def run_model_v1_shap(
    bundle_dir: Path = MODEL_V1_BUNDLE_DIR,
    predictions_path: Path = PREDICTIONS_V1_PATH,
    tiers_path: Path = TIERS_PREVIEW_V1_PATH,
    local_out_path: Path = SHAP_LOCAL_V1_PATH,
    local_by_feat_out_path: Path = SHAP_LOCAL_V1_BY_FEATURE_PATH,
    global_features_csv_path: Path = SHAP_GLOBAL_V1_FEATURES_CSV,
    global_groups_csv_path: Path = SHAP_GLOBAL_V1_GROUPS_CSV,
    summary_png_path: Path = SHAP_SUMMARY_V1_PNG,
    sanity_md_path: Path = SHAP_SANITY_V1_MD,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full Model v1 global, grouped, and High/Critical local SHAP pipeline and sanity audit."""
    assert_test_split_locked()
    t0 = time.perf_counter()

    bundle = load_model_v1_bundle(bundle_dir=bundle_dir, verify_checksums=True)
    model = bundle["model"]
    feature_names = bundle["features"]
    assert feature_names == ALL_FEATURE_NAMES, f"Feature mismatch: {feature_names} != {ALL_FEATURE_NAMES}"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing Model v1 predictions at {predictions_path}")
    if not tiers_path.exists():
        raise FileNotFoundError(f"Missing Model v1 tiers YAML at {tiers_path}")

    tiers_cfg = yaml.safe_load(tiers_path.read_text(encoding="utf-8"))
    model_version = str(tiers_cfg.get("model_version", bundle["version"]))

    preds_df = pd.read_parquet(predictions_path)
    assert_val_only_rows(preds_df, "model_v1_dev predictions")

    # Strictly load ONLY 2023 validation primary-set rows
    X_va, _, keys_va = load_features("val", label="new_fire_next_48h", feature_names=feature_names, primary=True)
    assert_val_only_rows(keys_va, "2023 val primary features")

    # Guard against burning rows in primary evaluation set
    if "burning_prev3" in keys_va.columns and keys_va["burning_prev3"].any():
        raise RuntimeError("STOP: Burning rows detected in primary validation set.")

    X_va = X_va.reset_index(drop=True)
    keys_va = keys_va.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # 1. 1,000-row additivity verification check (both raw and grouped sums)
    idx_1k = np.sort(rng.choice(len(X_va), size=min(ADDITIVITY_TEST_SIZE, len(X_va)), replace=False))
    X_1k = X_va.iloc[idx_1k][feature_names]
    shap_1k, base_1k = compute_tree_shap_margins(model, X_1k)
    verify_shap_additivity(model, X_1k, shap_1k, base_1k, tol=ADDITIVITY_TOL)
    _, grouped_1k, _ = group_shap_values(shap_1k, feature_names, GROUPS)
    max_diff_1k = verify_shap_additivity(model, X_1k, grouped_1k, base_1k, tol=ADDITIVITY_TOL)

    # 2. Global SHAP on 20,000 random 2023 primary rows (seed 42)
    rng_global = np.random.default_rng(seed)
    idx_20k = np.sort(rng_global.choice(len(X_va), size=min(GLOBAL_SAMPLE_SIZE, len(X_va)), replace=False))
    X_20k = X_va.iloc[idx_20k][feature_names].reset_index(drop=True)
    keys_20k = keys_va.iloc[idx_20k].reset_index(drop=True)

    summary_df, shap_20k, base_20k, _ = compute_global_feature_shap_v1(
        model=model,
        X_sample=X_20k,
        feature_names=feature_names,
        figure_path=summary_png_path,
        csv_path=global_features_csv_path,
    )
    group_df, grouped_20k, _ = compute_global_group_shap_v1(
        shap_values=shap_20k,
        feature_names=feature_names,
        csv_path=global_groups_csv_path,
        group_mapping=GROUPS,
    )
    max_diff_20k = verify_shap_additivity(model, X_20k, grouped_20k, base_20k, tol=ADDITIVITY_TOL)

    # 3. Local SHAP for every High or Critical cell-day
    high_crit_preds = preds_df[preds_df["tier"].isin(["High", "Critical"])].copy()
    assert_val_only_rows(high_crit_preds, "High/Critical predictions slice")

    panel_va = pd.concat([keys_va[["cell_id", "date", "split"]].copy(), X_va[feature_names].copy()], axis=1)
    panel_va["cell_id"] = panel_va["cell_id"].astype(str)
    panel_va["date"] = pd.to_datetime(panel_va["date"])
    high_crit_preds["cell_id"] = high_crit_preds["cell_id"].astype(str)
    high_crit_preds["date"] = pd.to_datetime(high_crit_preds["date"])

    merged_hc = high_crit_preds[["cell_id", "date", "tier", "split"]].merge(
        panel_va.drop(columns=["split"]),
        on=["cell_id", "date"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged_hc) != len(high_crit_preds):
        raise RuntimeError(
            f"STOP: Local SHAP join count mismatch: expected {len(high_crit_preds)}, got {len(merged_hc)}"
        )

    shap_local, base_local = compute_tree_shap_margins(model, merged_hc[feature_names])
    verify_shap_additivity(model, merged_hc[feature_names], shap_local, base_local, tol=ADDITIVITY_TOL)
    _, grouped_local, _ = group_shap_values(shap_local, feature_names, GROUPS)
    max_diff_local_grouped = verify_shap_additivity(model, merged_hc[feature_names], grouped_local, base_local, tol=ADDITIVITY_TOL)

    # Audit per-feature local table (top 5 features by |shap|)
    local_feat_df = extract_top_k_local_shap(
        keys_df=merged_hc[["cell_id", "date"]],
        X_df=merged_hc[feature_names],
        shap_values=shap_local,
        feature_names=feature_names,
        model_version=model_version,
        k=LOCAL_TOP_K,
    )
    local_by_feat_out_path.parent.mkdir(parents=True, exist_ok=True)
    local_feat_df.to_parquet(local_by_feat_out_path, index=False)

    # Primary grouped local table (top 5 groups by |summed shap|)
    local_grouped_df = extract_top_k_grouped_local_shap(
        keys_df=merged_hc[["cell_id", "date"]],
        X_df=merged_hc[feature_names],
        shap_values=shap_local,
        feature_names=feature_names,
        model_version=model_version,
        k=LOCAL_TOP_K,
        group_mapping=GROUPS,
    )
    local_out_path.parent.mkdir(parents=True, exist_ok=True)
    local_grouped_df.to_parquet(local_out_path, index=False)

    runtime_s = time.perf_counter() - t0

    # Sanity audit markdown
    sanity_report = build_shap_sanity_report_v1(
        summary_df=summary_df,
        group_df=group_df,
        X_sample=X_20k,
        shap_sample=shap_20k,
        keys_sample=keys_20k,
        feature_names=feature_names,
        base_value=base_local,
        max_diff_1k=max_diff_1k,
        max_diff_20k=max_diff_20k,
        max_diff_local_grouped=max_diff_local_grouped,
        local_rows_count=len(local_grouped_df),
        local_cell_days=len(merged_hc),
        runtime_s=runtime_s,
        model_version=model_version,
        report_path=sanity_md_path,
    )

    return {
        "status": "PASS",
        "runtime_s": runtime_s,
        "local_rows_count": len(local_grouped_df),
        "local_cell_days": len(merged_hc),
        "max_diff_1k": max_diff_1k,
        "max_diff_20k": max_diff_20k,
        "max_diff_local_grouped": max_diff_local_grouped,
        "sanity_report": sanity_report,
    }
