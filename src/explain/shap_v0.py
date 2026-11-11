"""Model v0 (xgb_v0_plain) SHAP global summary and High/Critical local explanations.

All SHAP contributions are computed in log-odds (model margin) units, NOT
probabilities, using shap.TreeExplainer (exact TreeSHAP for XGBoost). Strictly
reads only 2023 validation primary-set rows and fails loudly if any train/test
row or additivity violation (> 1e-4) occurs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from scipy.stats import spearmanr

from src.config import DOCS_AUDIT_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.explain.labels import FEATURE_LABELS, get_feature_label
from src.models.dev_predictions import MODEL_V0_PATH, PREDICTIONS_V0_PATH, TIERS_PREVIEW_V0_PATH
from src.models.features_io import load_features
from src.models.train_v0 import V0_FEATURES
from src.models.train_v1 import assert_test_split_locked

SHAP_LOCAL_V0_PATH = PROCESSED_DIR / "shap" / "model_v0_dev.parquet"
SHAP_GLOBAL_V0_CSV = DOCS_AUDIT_DIR / "shap_global_v0.csv"
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


def extract_top_k_local_shap(
    keys_df: pd.DataFrame,
    X_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
    model_version: str,
    k: int = LOCAL_TOP_K,
) -> pd.DataFrame:
    """Extract top-k features by |shap| per row into long-format parquet schema."""
    assert len(keys_df) == len(X_df) == shap_values.shape[0]
    assert shap_values.shape[1] == len(feature_names)

    n_rows = len(keys_df)
    top_idx = np.argsort(-np.abs(shap_values), axis=1, kind="stable")[:, :k]
    row_idx = np.repeat(np.arange(n_rows), k)
    feat_idx = top_idx.ravel()
    ranks = np.tile(np.arange(1, k + 1, dtype=np.int8), n_rows)

    x_mat = X_df[feature_names].to_numpy(dtype=np.float32)
    shap_f32 = shap_values.astype(np.float32)

    feat_arr = np.asarray(feature_names, dtype=object)[feat_idx]
    label_lookup = {f: get_feature_label(f) for f in feature_names}
    label_arr = np.asarray([label_lookup[f] for f in feat_arr], dtype=object)

    out_df = pd.DataFrame(
        {
            "cell_id": np.repeat(keys_df["cell_id"].astype(str).to_numpy(), k),
            "date": np.repeat(pd.to_datetime(keys_df["date"]).to_numpy(), k),
            "rank": ranks,
            "feature": feat_arr.astype(str),
            "label": label_arr.astype(str),
            "value": x_mat[row_idx, feat_idx].astype(np.float32),
            "shap": shap_f32[row_idx, feat_idx].astype(np.float32),
            "model_version": str(model_version),
        }
    )
    return out_df


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

    # Render beeswarm plot in headless Agg backend
    plt.switch_backend("Agg")
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
    X_sample: pd.DataFrame,
    shap_sample: np.ndarray,
    feature_names: list[str],
    base_value: float,
    max_diff_1k: float,
    max_diff_20k: float,
    max_diff_local: float,
    local_rows_count: int,
    local_cell_days: int,
    runtime_s: float,
    model_version: str,
    report_path: Path = SHAP_SANITY_V0_MD,
) -> str:
    """Write audit markdown assessing global rankings, directional fire logic, and dominance."""
    total_abs = float(summary_df["mean_abs_shap"].sum())
    table_rows = []
    for _, r in summary_df.iterrows():
        feat = str(r["feature"])
        share = (float(r["mean_abs_shap"]) / total_abs * 100.0) if total_abs > 0 else 0.0
        table_rows.append(
            f"| {int(r['rank'])} | `{feat}` | {get_feature_label(feat)} | {float(r['mean_abs_shap']):.6f} | {share:.2f}% |"
        )

    # Directional check for recent fire activity features
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

    # Location + Time-of-Year dominance check
    loc_feats = {"lat_c", "lon_c"}
    time_feats = {"month_sin", "month_cos"}
    fire_feats = {"fire_count_7d", "fire_count_30d", "neighbour_fire_7d"}
    loc_abs = float(summary_df.loc[summary_df["feature"].isin(loc_feats), "mean_abs_shap"].sum())
    time_abs = float(summary_df.loc[summary_df["feature"].isin(time_feats), "mean_abs_shap"].sum())
    fire_abs = float(summary_df.loc[summary_df["feature"].isin(fire_feats), "mean_abs_shap"].sum())
    loc_pct = loc_abs / total_abs * 100.0 if total_abs > 0 else 0.0
    time_pct = time_abs / total_abs * 100.0 if total_abs > 0 else 0.0
    fire_pct = fire_abs / total_abs * 100.0 if total_abs > 0 else 0.0
    loc_time_pct = loc_pct + time_pct

    top_rank_feat = str(summary_df.iloc[0]["feature"])
    top2_feats = set(summary_df.head(2)["feature"].astype(str))
    dominates = (loc_time_pct > 55.0) or (top_rank_feat in (loc_feats | time_feats) and top2_feats <= (loc_feats | time_feats))

    lines = [
        "# Model v0 SHAP Sanity Audit",
        "",
        f"- **Model version:** `{model_version}` (`xgb_v0_plain`, 7 base features, no weather)",
        f"- **Route used:** `shap=={shap.__version__}` (`shap.TreeExplainer`, exact TreeSHAP)",
        "- **Units:** All SHAP values (`shap`) and base value are in **log-odds (margin) units**, NOT probabilities.",
        f"- **Expected (base) value:** `{base_value:.6f}` log-odds",
        f"- **Additivity check (`|sum(shap) + base - margin|`):** max diff `{max_diff_1k:.2e}` (1,000 random rows), "
        f"`{max_diff_20k:.2e}` (20,000 global rows), `{max_diff_local:.2e}` ({local_cell_days:,} High/Critical local rows) — all <= `1e-4` (**PASS**).",
        f"- **Scope verified:** Strictly 2023 validation primary-set rows (`split == \"val\"`). Local parquet (`{local_rows_count:,}` rows = `{local_cell_days:,}` cell-days x top 5) computed in `{runtime_s:.2f}s`.",
        "",
        "## 1. Global Feature Ranking (20,000 Random 2023 Primary Rows, Seed 42)",
        "",
        "| Rank | Feature | Display Label | Mean \\|SHAP\\| (log-odds) | Share |",
        "|---|---|---|---|---|",
        *table_rows,
        "",
        "## 2. Do Recent Fires Increase Predicted Risk?",
        "",
        *fire_checks,
        "",
        "## 3. Location and Seasonality Dominance Check",
        "",
        f"- **Location (`lat_c` + `lon_c`) share:** `{loc_pct:.2f}%` (sum mean |SHAP| = `{loc_abs:.4f}`)",
        f"- **Time of year (`month_sin` + `month_cos`) share:** `{time_pct:.2f}%` (sum mean |SHAP| = `{time_abs:.4f}`)",
        f"- **Combined Location + Time-of-Year share:** `{loc_time_pct:.2f}%` vs **Recent Fire History share:** `{fire_pct:.2f}%`.",
        "",
    ]

    if dominates:
        lines.extend(
            [
                "> **FINDING (Static Space-Time Dominance in Model v0):** Location (`lat_c`, `lon_c`) and/or Time of Year "
                f"(`month_sin`, `month_cos`) dominate Model v0 (`{loc_time_pct:.2f}%` combined importance; rank #1 is `{top_rank_feat}`). "
                "Because Model v0 has no weather or drought inputs yet, it relies heavily on geographic coordinates and seasonality as proxies for seasonal fire climatology. "
                "We expect dynamic weather lags and KBDI in Model v1 to replace part of this static spatial/seasonal reliance.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"- **Finding:** Top rank is `{top_rank_feat}` (`{fire_pct:.2f}%` fire-history share vs `{loc_time_pct:.2f}%` location + time-of-year share); static priors do not overwhelm recent fire activity.",
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
    global_csv_path: Path = SHAP_GLOBAL_V0_CSV,
    summary_png_path: Path = SHAP_SUMMARY_V0_PNG,
    sanity_md_path: Path = SHAP_SANITY_V0_MD,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full Model v0 global & High/Critical local SHAP pipeline and audit checks."""
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

    # Reset index so integer slicing aligns cleanly
    X_va = X_va.reset_index(drop=True)
    keys_va = keys_va.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Explicit 1,000-row additivity verification check
    idx_1k = np.sort(rng.choice(len(X_va), size=min(ADDITIVITY_TEST_SIZE, len(X_va)), replace=False))
    shap_1k, base_1k = compute_tree_shap_margins(model, X_va.iloc[idx_1k][V0_FEATURES])
    max_diff_1k = verify_shap_additivity(model, X_va.iloc[idx_1k][V0_FEATURES], shap_1k, base_1k)

    # Global SHAP on 20,000 random 2023 primary rows (seed 42)
    rng_global = np.random.default_rng(seed)
    idx_20k = np.sort(rng_global.choice(len(X_va), size=min(GLOBAL_SAMPLE_SIZE, len(X_va)), replace=False))
    X_20k = X_va.iloc[idx_20k][V0_FEATURES].reset_index(drop=True)
    summary_df, shap_20k, base_20k, max_diff_20k = compute_global_shap_summary(
        model=model,
        X_sample=X_20k,
        feature_names=V0_FEATURES,
        figure_path=summary_png_path,
        csv_path=global_csv_path,
    )

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
    max_diff_local = verify_shap_additivity(model, merged_hc[V0_FEATURES], shap_local, base_local)

    local_df = extract_top_k_local_shap(
        keys_df=merged_hc[["cell_id", "date"]],
        X_df=merged_hc[V0_FEATURES],
        shap_values=shap_local,
        feature_names=V0_FEATURES,
        model_version=model_version,
        k=LOCAL_TOP_K,
    )
    local_out_path.parent.mkdir(parents=True, exist_ok=True)
    local_df.to_parquet(local_out_path, index=False)

    runtime_s = time.perf_counter() - t0
    build_shap_sanity_report(
        summary_df=summary_df,
        X_sample=X_20k,
        shap_sample=shap_20k,
        feature_names=V0_FEATURES,
        base_value=base_20k,
        max_diff_1k=max_diff_1k,
        max_diff_20k=max_diff_20k,
        max_diff_local=max_diff_local,
        local_rows_count=len(local_df),
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
        "max_diff_local": max_diff_local,
        "local_cell_days": len(merged_hc),
        "local_rows": len(local_df),
        "runtime_seconds": round(runtime_s, 2),
        "global_summary": summary_df,
    }

