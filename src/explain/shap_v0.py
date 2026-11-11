"""Model v0 (xgb_v0_plain) SHAP global summary and grouped High/Critical local explanations.

All SHAP contributions are computed in log-odds (model margin) units, NOT
probabilities, using shap.TreeExplainer (exact TreeSHAP for XGBoost). Raw
per-feature SHAP attributions are summed within each display group (`GROUPS`)
for operational clarity while keeping exact per-feature attributions as an audit
table. Strictly reads only 2023 validation primary-set rows and fails loudly if
any train/test row or additivity violation (> 1e-4) occurs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from scipy.stats import spearmanr

from src.config import DOCS_AUDIT_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.explain.labels import FEATURE_LABELS, GROUPS, get_driver_group, get_feature_label
from src.models.dev_predictions import MODEL_V0_PATH, PREDICTIONS_V0_PATH, TIERS_PREVIEW_V0_PATH
from src.models.features_io import load_features
from src.models.train_v0 import V0_FEATURES
from src.models.train_v1 import assert_test_split_locked
from src.utils.plotting import plt

SHAP_LOCAL_V0_PATH = PROCESSED_DIR / "shap" / "model_v0_dev.parquet"
SHAP_LOCAL_V0_BY_FEATURE_PATH = PROCESSED_DIR / "shap" / "model_v0_dev_by_feature.parquet"
SHAP_GLOBAL_V0_CSV = DOCS_AUDIT_DIR / "shap_global_v0.csv"
SHAP_GLOBAL_V0_GROUPS_CSV = DOCS_AUDIT_DIR / "shap_global_v0_groups.csv"
SHAP_SUMMARY_V0_PNG = PROJECT_ROOT / "docs" / "figures" / "shap_summary_v0.png"
SHAP_SANITY_V0_MD = DOCS_AUDIT_DIR / "shap_sanity_v0.md"

ADDITIVITY_TOL = 1e-4
GLOBAL_SAMPLE_SIZE = 20_000
ADDITIVITY_TEST_SIZE = 1_000
LOCAL_TOP_K = 5


def compute_tree_shap_margins(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Compute exact TreeSHAP values and scalar base value in log-odds margin units."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0], dtype=np.float64)
    else:
        shap_arr = np.asarray(shap_values, dtype=np.float64)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1]

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev_arr = np.asarray(ev).ravel()
        base_value = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
    else:
        base_value = float(ev)

    return shap_arr, base_value


def verify_shap_additivity(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    base_value: float,
    tol: float = ADDITIVITY_TOL,
) -> float:
    """Verify sum(shap) + base_value == model log-odds margin within `tol`; STOP on failure."""
    margins = np.asarray(model.predict(X, output_margin=True), dtype=np.float64).ravel()
    reconstructed = np.asarray(shap_values, dtype=np.float64).sum(axis=1) + float(base_value)
    max_diff = float(np.max(np.abs(reconstructed - margins)))
    if not np.isfinite(max_diff) or max_diff > tol:
        raise RuntimeError(f"STOP: SHAP additivity failed (max |sum(shap)+base-margin| = {max_diff:.6e} > {tol})")
    return max_diff


def assert_val_only_rows(df: pd.DataFrame, context: str) -> None:
    """Fail loudly if any non-validation (train, test, gap, or non-2023) row is present."""
    if "split" in df.columns:
        bad_splits = set(df["split"].astype(str).unique()) - {"val"}
        if bad_splits:
            raise RuntimeError(f"STOP: Non-val split(s) {bad_splits} detected in {context}.")
    years = pd.to_datetime(df["date"]).dt.year.unique()
    if set(years.tolist()) != {2023}:
        raise RuntimeError(f"STOP: Non-2023 year(s) {years.tolist()} detected in {context}.")


def group_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
    group_mapping: dict[str, str] | None = None,
) -> tuple[list[str], np.ndarray, list[list[str]]]:
    """Sum per-feature SHAP values within each driver group in deterministic order."""
    mapping = group_mapping or GROUPS
    driver_names: list[str] = []
    member_cols: dict[str, list[int]] = {}
    member_names: dict[str, list[str]] = {}

    for j, feat in enumerate(feature_names):
        drv = mapping[feat] if feat in mapping else get_driver_group(feat)
        if drv not in member_cols:
            driver_names.append(drv)
            member_cols[drv] = []
            member_names[drv] = []
        member_cols[drv].append(j)
        member_names[drv].append(feat)

    grouped = np.column_stack([shap_values[:, member_cols[drv]].sum(axis=1) for drv in driver_names]).astype(np.float64)
    members = [member_names[drv] for drv in driver_names]
    return driver_names, grouped, members


def _format_scalar(v: float) -> str:
    clean = 0.0 if abs(float(v)) < 1e-12 else round(float(v), 4)
    return f"{clean:g}"


def extract_top_k_local_shap(
    keys_df: pd.DataFrame,
    X_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    model_version: str,
    k: int = LOCAL_TOP_K,
) -> pd.DataFrame:
    """Extract top-k raw features by |shap| per row into long-format audit parquet schema."""
    assert len(keys_df) == len(X_df) == shap_values.shape[0]
    assert shap_values.shape[1] == len(feature_names)

    top_k = min(k, len(feature_names))
    n_rows = len(keys_df)
    top_idx = np.argsort(-np.abs(shap_values), axis=1, kind="stable")[:, :top_k]
    row_idx = np.repeat(np.arange(n_rows), top_k)
    feat_idx = top_idx.ravel()
    ranks = np.tile(np.arange(1, top_k + 1, dtype=np.int8), n_rows)

    x_mat = X_df[feature_names].to_numpy(dtype=np.float32)
    shap_f32 = shap_values.astype(np.float32)

    feat_arr = np.asarray(feature_names, dtype=object)[feat_idx]
    label_lookup = {f: get_feature_label(f) for f in feature_names}
    label_arr = np.asarray([label_lookup[f] for f in feat_arr], dtype=object)

    return pd.DataFrame(
        {
            "cell_id": np.repeat(keys_df["cell_id"].astype(str).to_numpy(), top_k),
            "date": np.repeat(pd.to_datetime(keys_df["date"]).to_numpy(), top_k),
            "rank": ranks,
            "feature": feat_arr.astype(str),
            "label": label_arr.astype(str),
            "value": x_mat[row_idx, feat_idx].astype(np.float32),
            "shap": shap_f32[row_idx, feat_idx].astype(np.float32),
            "model_version": str(model_version),
        }
    )


def extract_top_k_grouped_local_shap(
    keys_df: pd.DataFrame,
    X_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    model_version: str,
    k: int = LOCAL_TOP_K,
    group_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Sum SHAP per group and keep top-k driver groups by |summed shap| per cell-day."""
    assert len(keys_df) == len(X_df) == shap_values.shape[0]
    assert shap_values.shape[1] == len(feature_names)

    driver_names, grouped_shap, group_members = group_shap_values(shap_values, feature_names, group_mapping)
    top_k = min(k, len(driver_names))
    n_rows = len(keys_df)

    top_idx = np.argsort(-np.abs(grouped_shap), axis=1, kind="stable")[:, :top_k]
    row_idx = np.repeat(np.arange(n_rows), top_k)
    grp_idx = top_idx.ravel()
    ranks = np.tile(np.arange(1, top_k + 1, dtype=np.int8), n_rows)

    grouped_f32 = grouped_shap.astype(np.float32)
    drv_arr = np.asarray(driver_names, dtype=object)[grp_idx]

    # Precompute feature value matrix for fast string formatting
    col_arrays = {f: X_df[f].to_numpy(dtype=np.float64) for f in feature_names}
    details: list[str] = []
    for r_i, g_i in zip(row_idx, grp_idx):
        members = group_members[int(g_i)]
        details.append(", ".join(f"{m}={_format_scalar(col_arrays[m][int(r_i)])}" for m in members))

    return pd.DataFrame(
        {
            "cell_id": np.repeat(keys_df["cell_id"].astype(str).to_numpy(), top_k),
            "date": np.repeat(pd.to_datetime(keys_df["date"]).to_numpy(), top_k),
            "rank": ranks,
            "driver": drv_arr.astype(str),
            "shap": grouped_f32[row_idx, grp_idx],
            "feature_detail": pd.Series(details, dtype="str"),
            "model_version": str(model_version),
        }
    )


def compute_global_group_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    csv_path: Path = SHAP_GLOBAL_V0_GROUPS_CSV,
    group_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Compute global driver importance as mean(|summed group SHAP|) over sampled rows."""
    driver_names, grouped_shap, _ = group_shap_values(shap_values, feature_names, group_mapping)
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


def compute_global_shap_summary(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    figure_path: Path = SHAP_SUMMARY_V0_PNG,
    csv_path: Path = SHAP_GLOBAL_V0_CSV,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    """Compute global SHAP ranking on 20k sample, check additivity, and save CSV + summary plot."""
    shap_vals, base_val = compute_tree_shap_margins(model, X_sample[feature_names])
    max_diff = verify_shap_additivity(model, X_sample[feature_names], shap_vals, base_val)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    summary_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs.astype(float)})
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
    summary_df["rank"] = np.arange(1, len(summary_df) + 1, dtype=int)
    summary_df["mean_abs_shap"] = summary_df["mean_abs_shap"].round(6)
    summary_df = summary_df[["feature", "mean_abs_shap", "rank"]]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)

    # Render beeswarm plot using shared Agg backend
    display_names = [f"{f} ({FEATURE_LABELS[f]})" for f in feature_names]
    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_vals, X_sample[feature_names], feature_names=display_names, show=False)
    plt.title("Model v0 Global SHAP Summary (20,000 Random 2023 Primary Rows, Log-Odds Units)")
    plt.xlabel("SHAP value (impact on log-odds margin)")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    return summary_df, shap_vals, base_val, max_diff


def build_shap_sanity_report(
    summary_df: pd.DataFrame,
    group_df: pd.DataFrame,
    X_sample: pd.DataFrame,
    shap_sample: np.ndarray,
    feature_names: list[str],
    base_value: float,
    max_diff_1k: float,
    max_diff_20k: float,
    max_diff_local_grouped: float,
    local_rows_count: int,
    local_cell_days: int,
    runtime_s: float,
    model_version: str,
    report_path: Path = SHAP_SANITY_V0_MD,
) -> str:
    """Write audit markdown assessing grouped & per-feature rankings, fire direction, and dominance."""
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
            f"| {int(r['rank'])} | `{feat}` | {get_driver_group(feat)} | {float(r['mean_abs_shap']):.6f} | {share:.2f}% |"
        )

    fire_checks = []
    for f in ["fire_count_7d", "fire_count_30d", "neighbour_fire_7d"]:
        j = feature_names.index(f)
        vals = X_sample[f].to_numpy(dtype=np.float64)
        sh = shap_sample[:, j]
        pos_mask = vals > 0
        zero_mask = vals == 0
        mean_pos = float(sh[pos_mask].mean()) if pos_mask.any() else float("nan")
        mean_zero = float(sh[zero_mask].mean()) if zero_mask.any() else float("nan")
        corr_res = spearmanr(vals, sh)
        rho = float(corr_res.statistic)
        fire_checks.append(
            f"- `{f}` ({get_feature_label(f)}): mean SHAP when `> 0` is `{mean_pos:+.4f}` "
            f"(n={int(pos_mask.sum()):,}) vs `== 0` `{mean_zero:+.4f}` (n={int(zero_mask.sum()):,}); "
            f"Spearman `rho = {rho:+.4f}`."
        )

    reg_abs = float(group_df.loc[group_df["driver"] == "Regional baseline", "mean_abs_shap"].sum())
    seas_abs = float(group_df.loc[group_df["driver"] == "Seasonal baseline", "mean_abs_shap"].sum())
    fire_grp_names = {
        FEATURE_LABELS["fire_count_7d"],
        FEATURE_LABELS["fire_count_30d"],
        FEATURE_LABELS["neighbour_fire_7d"],
    }
    fire_grp_abs = float(group_df.loc[group_df["driver"].isin(fire_grp_names), "mean_abs_shap"].sum())
    reg_pct = reg_abs / total_grp_abs * 100.0 if total_grp_abs > 0 else 0.0
    seas_pct = seas_abs / total_grp_abs * 100.0 if total_grp_abs > 0 else 0.0
    fire_grp_pct = fire_grp_abs / total_grp_abs * 100.0 if total_grp_abs > 0 else 0.0
    baseline_pct = reg_pct + seas_pct

    top_drv = str(group_df.iloc[0]["driver"])
    dominates = top_drv in {"Regional baseline", "Seasonal baseline"} or baseline_pct > 55.0

    lines = [
        "# Model v0 SHAP Sanity Audit",
        "",
        f"- **Model version:** `{model_version}` (`xgb_v0_plain`, 7 base features grouped into 5 drivers)",
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
        "## 2. Per-Feature Audit Ranking (`docs/audit/shap_global_v0.csv`)",
        "",
        "| Rank | Feature | Driver Group | Mean \\|SHAP\\| (log-odds) | Share |",
        "|---|---|---|---|---|",
        *feat_table_rows,
        "",
        "## 3. Do Recent Fires Increase Predicted Risk?",
        "",
        *fire_checks,
        "",
        "## 4. Regional & Seasonal Baseline Dominance Check (Grouped)",
        "",
        f"- **Regional baseline (`lat_c + lon_c`) share:** `{reg_pct:.2f}%` (mean |summed SHAP| = `{reg_abs:.4f}`)",
        f"- **Seasonal baseline (`month_sin + month_cos`) share:** `{seas_pct:.2f}%` (mean |summed SHAP| = `{seas_abs:.4f}`)",
        f"- **Combined baseline share:** `{baseline_pct:.2f}%` vs **Combined recent fire activity share:** `{fire_grp_pct:.2f}%`.",
        "",
    ]

    if dominates:
        lines.extend(
            [
                "> **FINDING (Static Regional/Seasonal Baseline Dominance in Grouped Model v0 SHAP):** "
                f"`{top_drv}` ranks #1 at grouped driver level and combined `Regional baseline` + `Seasonal baseline` accounts for "
                f"`{baseline_pct:.2f}%` of grouped mean |SHAP| (`{fire_grp_pct:.2f}%` from fire counts). "
                "Because `xgb_v0_plain` lacks dynamic weather and drought inputs, geographic coordinates (`lat_c + lon_c`) and calendar seasonality (`month_sin + month_cos`) act together as the model's largest single risk prior.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- **Finding:** Top grouped driver is `{top_drv}`; `Regional baseline` (`{reg_pct:.2f}%`) and `Seasonal baseline` (`{seas_pct:.2f}%`) do not dominate everything.",
                "",
            ]
        )

    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text


def run_model_v0_shap(
    model_path: Path = MODEL_V0_PATH,
    predictions_path: Path = PREDICTIONS_V0_PATH,
    tiers_path: Path = TIERS_PREVIEW_V0_PATH,
    local_out_path: Path = SHAP_LOCAL_V0_PATH,
    local_by_feat_out_path: Path = SHAP_LOCAL_V0_BY_FEATURE_PATH,
    global_csv_path: Path = SHAP_GLOBAL_V0_CSV,
    global_groups_csv_path: Path = SHAP_GLOBAL_V0_GROUPS_CSV,
    summary_png_path: Path = SHAP_SUMMARY_V0_PNG,
    sanity_md_path: Path = SHAP_SANITY_V0_MD,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full Model v0 global, grouped, and High/Critical local SHAP pipeline and audit checks."""
    assert_test_split_locked()
    t0 = time.perf_counter()

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained Model v0 at {model_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing Model v0 predictions at {predictions_path}")
    if not tiers_path.exists():
        raise FileNotFoundError(f"Missing Model v0 tiers YAML at {tiers_path}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    tiers_cfg = yaml.safe_load(tiers_path.read_text(encoding="utf-8"))
    model_version = str(tiers_cfg["model_version"])

    preds_df = pd.read_parquet(predictions_path)
    assert_val_only_rows(preds_df, "model_v0_dev predictions")

    X_va, _, keys_va = load_features("val", label="new_fire_next_48h", feature_names=V0_FEATURES, primary=True)
    assert_val_only_rows(keys_va, "2023 val primary features")

    X_va = X_va.reset_index(drop=True)
    keys_va = keys_va.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Explicit 1,000-row additivity verification check (both raw and grouped sums)
    idx_1k = np.sort(rng.choice(len(X_va), size=min(ADDITIVITY_TEST_SIZE, len(X_va)), replace=False))
    X_1k = X_va.iloc[idx_1k][V0_FEATURES]
    shap_1k, base_1k = compute_tree_shap_margins(model, X_1k)
    verify_shap_additivity(model, X_1k, shap_1k, base_1k)
    _, grouped_1k, _ = group_shap_values(shap_1k, V0_FEATURES)
    max_diff_1k = verify_shap_additivity(model, X_1k, grouped_1k, base_1k)

    # Global SHAP on 20,000 random 2023 primary rows (seed 42)
    rng_global = np.random.default_rng(seed)
    idx_20k = np.sort(rng_global.choice(len(X_va), size=min(GLOBAL_SAMPLE_SIZE, len(X_va)), replace=False))
    X_20k = X_va.iloc[idx_20k][V0_FEATURES].reset_index(drop=True)
    summary_df, shap_20k, base_20k, _ = compute_global_shap_summary(
        model=model,
        X_sample=X_20k,
        feature_names=V0_FEATURES,
        figure_path=summary_png_path,
        csv_path=global_csv_path,
    )
    group_df, grouped_20k, _ = compute_global_group_shap(
        shap_values=shap_20k,
        feature_names=V0_FEATURES,
        csv_path=global_groups_csv_path,
    )
    max_diff_20k = verify_shap_additivity(model, X_20k, grouped_20k, base_20k)

    # Local SHAP for every High or Critical cell-day
    high_crit_preds = preds_df[preds_df["tier"].isin(["High", "Critical"])].copy()
    assert_val_only_rows(high_crit_preds, "High/Critical predictions slice")

    panel_va = pd.concat([keys_va[["cell_id", "date", "split"]].copy(), X_va[V0_FEATURES].copy()], axis=1)
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

    shap_local, base_local = compute_tree_shap_margins(model, merged_hc[V0_FEATURES])
    verify_shap_additivity(model, merged_hc[V0_FEATURES], shap_local, base_local)
    _, grouped_local, _ = group_shap_values(shap_local, V0_FEATURES)
    max_diff_local_grouped = verify_shap_additivity(model, merged_hc[V0_FEATURES], grouped_local, base_local)

    # Audit per-feature local table
    local_feat_df = extract_top_k_local_shap(
        keys_df=merged_hc[["cell_id", "date"]],
        X_df=merged_hc[V0_FEATURES],
        shap_values=shap_local,
        feature_names=V0_FEATURES,
        model_version=model_version,
        k=LOCAL_TOP_K,
    )
    local_by_feat_out_path.parent.mkdir(parents=True, exist_ok=True)
    local_feat_df.to_parquet(local_by_feat_out_path, index=False)

    # Primary grouped local table
    local_grouped_df = extract_top_k_grouped_local_shap(
        keys_df=merged_hc[["cell_id", "date"]],
        X_df=merged_hc[V0_FEATURES],
        shap_values=shap_local,
        feature_names=V0_FEATURES,
        model_version=model_version,
        k=LOCAL_TOP_K,
    )
    local_out_path.parent.mkdir(parents=True, exist_ok=True)
    local_grouped_df.to_parquet(local_out_path, index=False)

    runtime_s = time.perf_counter() - t0
    build_shap_sanity_report(
        summary_df=summary_df,
        group_df=group_df,
        X_sample=X_20k,
        shap_sample=shap_20k,
        feature_names=V0_FEATURES,
        base_value=base_20k,
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
        "route": f"shap=={shap.__version__} (TreeExplainer, log-odds margin units)",
        "model_version": model_version,
        "base_value": base_20k,
        "max_diff_1k": max_diff_1k,
        "max_diff_20k": max_diff_20k,
        "max_diff_local_grouped": max_diff_local_grouped,
        "local_cell_days": len(merged_hc),
        "local_rows": len(local_grouped_df),
        "local_by_feature_rows": len(local_feat_df),
        "runtime_seconds": round(runtime_s, 2),
        "global_summary": summary_df,
        "global_groups": group_df,
    }
