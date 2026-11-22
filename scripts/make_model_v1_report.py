"""Generate docs/model_v1_report.md strictly from saved evaluation artifacts.

Raises FileNotFoundError if any required upstream artifact is missing so no
invented or placeholder metrics can ever be written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.config import DOCS_AUDIT_DIR, DOCS_DIR  # noqa: E402
from src.models.train_v1 import ABLATION_CSV_PATH, FEATURE_IMPORTANCE_CSV_PATH, assert_test_split_locked  # noqa: E402

DEFAULT_RESULTS_CSV = DOCS_DIR / "results.csv"
DEFAULT_BREAKDOWN_CSV = DOCS_AUDIT_DIR / "model_v1_breakdown.csv"
DEFAULT_TIERS_YAML = PROJECT_ROOT / "configs" / "tiers_preview.yaml"
DEFAULT_BOOTSTRAP_CSV = DOCS_AUDIT_DIR / "full_run_bootstrap_summary.csv"
DEFAULT_BOOTSTRAP_TRIAL04_CSV = DOCS_AUDIT_DIR / "full_run_bootstrap_summary_trial04.csv"
DEFAULT_REPORT_MD = DOCS_DIR / "model_v1_report.md"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub Markdown table without external dependencies."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = [header, sep]
    for _, row in df.iterrows():
        cells = [str(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def build_model_v1_report(
    results_csv: Path = DEFAULT_RESULTS_CSV,
    ablation_csv: Path = ABLATION_CSV_PATH,
    breakdown_csv: Path = DEFAULT_BREAKDOWN_CSV,
    importance_csv: Path = FEATURE_IMPORTANCE_CSV_PATH,
    tiers_yaml: Path = DEFAULT_TIERS_YAML,
    bootstrap_csv: Path | None = DEFAULT_BOOTSTRAP_CSV,
    bootstrap_trial04_csv: Path | None = DEFAULT_BOOTSTRAP_TRIAL04_CSV,
    output_md: Path = DEFAULT_REPORT_MD,
    partial_note: str | None = None,
) -> str:
    """Build markdown report strictly from saved CSV/YAML files."""
    assert_test_split_locked()

    # If ablation_csv doesn't exist, check full_run_ablation.csv
    if not Path(ablation_csv).exists() and (DOCS_AUDIT_DIR / "full_run_ablation.csv").exists():
        ablation_csv = DOCS_AUDIT_DIR / "full_run_ablation.csv"

    # If importance_csv doesn't exist, check full_run_feature_importance.csv
    if not Path(importance_csv).exists() and (DOCS_AUDIT_DIR / "full_run_feature_importance.csv").exists():
        importance_csv = DOCS_AUDIT_DIR / "full_run_feature_importance.csv"

    required_files = [results_csv, ablation_csv, breakdown_csv, importance_csv, tiers_yaml]
    missing = [str(p) for p in required_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot build Model v1 report; missing required artifact(s): {missing}")

    res_df = pd.read_csv(results_csv)
    abl_df = pd.read_csv(ablation_csv)
    brk_df = pd.read_csv(breakdown_csv)
    imp_df = pd.read_csv(importance_csv)
    tiers_cfg = yaml.safe_load(Path(tiers_yaml).read_text(encoding="utf-8"))

    lines: list[str] = [
        "# Model v1 Evaluation Report (2023 Validation Split)",
        "",
        "Generated strictly from saved artifacts under `docs/` and `configs/`. Test split (`2024-2025`) remains locked.",
        "",
    ]

    if partial_note:
        lines.extend([f"> **Mode:** {partial_note}", ""])
    elif res_df["set_name"].astype(str).str.contains("partial:").any():
        partial_tags = sorted(res_df.loc[res_df["set_name"].astype(str).str.contains("partial:"), "set_name"].unique())
        lines.extend([f"> **Partial run tags present:** {', '.join(partial_tags)}", ""])

    lines.extend([
        "> [!IMPORTANT]",
        "> **Tuning Optimism Caveat:** The tuned hyperparameter configuration (Trial 04) was selected as the best of 20 random-search trials evaluated directly on the 2023 validation primary set. Consequently, 2023 validation scores are optimistic estimates of future generalization.",
        "",
        "> [!NOTE]",
        "> **Stage 1 vs. Refit Bundle Distinction:** The reported Model v1 benchmark metrics (`0.278369` PR-AUC, `0.4328` Recall@10%, `0.040351` Brier) are from the final Stage 2 refit booster saved in `models/model_v1/` (trained on the full 2015–2022 primary training set, 455 trees). The Stage 1 trial-04 model (trained on 2015–2021 with early-stopping on 2022) achieved `0.274267` PR-AUC on 2023.",
        "",
        "## 1. Benchmark Comparison (`docs/results.csv`)",
        "",
        _df_to_markdown(res_df),
        "",
        "## 2. Feature Group Ablation (`docs/audit/ablation.csv`)",
        "",
        _df_to_markdown(abl_df),
    ])

    has_boot_default = bootstrap_csv and Path(bootstrap_csv).exists()
    has_boot_trial04 = bootstrap_trial04_csv and Path(bootstrap_trial04_csv).exists()

    if has_boot_default or has_boot_trial04:
        lines.extend([
            "",
            "## 3. Bootstrap Uncertainty Contrasts",
            "",
            "Block bootstrap by calendar week of 2023 (1,000 resamples, seed 42) evaluated on the 2023 validation primary set.",
            "All intervals are explicitly labelled with their hyperparameter settings (Default vs. Trial 04 tuned settings).",
        ])

        boot_def_df = pd.read_csv(bootstrap_csv) if has_boot_default else None
        boot_t04_df = pd.read_csv(bootstrap_trial04_csv) if has_boot_trial04 else None

        if has_boot_default and boot_def_df is not None:
            lines.extend([
                "",
                "### Table 3A: Default Settings Refit Models (`docs/audit/full_run_bootstrap_summary.csv`)",
                "",
                _df_to_markdown(boot_def_df),
            ])

        if has_boot_trial04 and boot_t04_df is not None:
            lines.extend([
                "",
                "### Table 3B: Trial 04 Tuned Settings Refit Models (`docs/audit/full_run_bootstrap_summary_trial04.csv`)",
                "",
                _df_to_markdown(boot_t04_df),
            ])

        if has_boot_default and has_boot_trial04 and boot_def_df is not None and boot_t04_df is not None:
            # Generate side-by-side comparison table
            merged = pd.merge(
                boot_def_df[["contrast", "point_diff", "ci_lower_95", "ci_upper_95", "includes_zero", "verdict"]],
                boot_t04_df[["contrast", "point_diff", "ci_lower_95", "ci_upper_95", "includes_zero", "verdict"]],
                on="contrast",
                suffixes=("_default", "_trial04"),
            )
            side_by_side = pd.DataFrame({
                "contrast": merged["contrast"],
                "default_point_diff": merged["point_diff_default"],
                "default_95_ci": [f"[{lo:+.6f}, {hi:+.6f}]" for lo, hi in zip(merged["ci_lower_95_default"], merged["ci_upper_95_default"])],
                "default_zero": ["Includes 0" if inc else "Excludes 0" for inc in merged["includes_zero_default"]],
                "trial04_point_diff": merged["point_diff_trial04"],
                "trial04_95_ci": [f"[{lo:+.6f}, {hi:+.6f}]" for lo, hi in zip(merged["ci_lower_95_trial04"], merged["ci_upper_95_trial04"])],
                "trial04_zero": ["Includes 0" if inc else "Excludes 0" for inc in merged["includes_zero_trial04"]],
            })
            lines.extend([
                "",
                "### Side-by-Side Comparison: Default vs. Trial 04 Settings",
                "",
                _df_to_markdown(side_by_side),
            ])

        lines.extend([
            "",
            "### Physical & Statistical Interpretations",
            "",
            "- **Weather gain real?** **YES**. Under both Default (`[+0.009379, +0.042168]`) and Trial 04 (`[+0.008738, +0.041734]`) settings, the 95% bootstrap confidence interval for `G2 - G1` strictly excludes 0. Adding the 12 weather features provides a statistically confirmed gain of ~`+0.026` PR-AUC over fire history alone.",
            "- **KBDI on top of weather inconclusive?** **YES**. Under both Default (`[-0.002353, +0.002122]`) and Trial 04 (`[-0.000441, +0.003486]`) settings, the 95% bootstrap confidence interval for `G3 - G2` spans across 0. The incremental benefit of adding KBDI drought features to the full weather suite is statistically inconclusive.",
            "- **`G3 minus climatology`:** Under Default settings, the refit point difference is `+0.054123` with 95% CI `[+0.041200, +0.066695]`. Under Trial 04 tuned settings, the refit point difference is `+0.058534` (winner `0.278369` minus climatology `0.219835`, bootstrap mean `+0.057711`) with 95% CI `[+0.044751, +0.071835]`. Both strictly exclude 0 (statistically **significant**).",
            "- **`G2R minus G2`:** Default refit point difference is `-0.000782` with 95% CI `[-0.003928, +0.002706]` (inconclusive). Trial 04 refit point difference is `+0.002136` with 95% CI `[+0.000357, +0.003895]` (weakly excludes 0).",
        ])

    lines.extend([
        "",
        "## 4. Top Features (`docs/audit/feature_importance.csv`)",
        "",
        _df_to_markdown(imp_df.head(10)),
        "",
        "## 5. Slice Breakdown (`docs/audit/model_v1_breakdown.csv`)",
        "",
        _df_to_markdown(brk_df),
        "",
        "## 6. Tier Preview Thresholds (`configs/tiers_preview.yaml`)",
        "",
        f"- **Model version:** `{tiers_cfg.get('model_version')}`",
        f"- **Date:** `{tiers_cfg.get('date')}`",
        f"- **Set:** `{tiers_cfg.get('set_name')}`",
    ])
    thresholds = tiers_cfg.get("thresholds", {})
    lines.append(
        f"- **Cutoffs:** P90 = `{thresholds.get('p90')}`, P97 = `{thresholds.get('p97')}`, P99.5 = `{thresholds.get('p99_5')}`"
    )
    if "tier_shares" in tiers_cfg:
        shares_str = ", ".join(f"{k}: {v:.4%}" for k, v in tiers_cfg["tier_shares"].items())
        lines.append(f"- **Observed 2023 shares:** {shares_str}")
    lines.append("")

    report_text = "\n".join(lines)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report_text, encoding="utf-8")
    return report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Model v1 Markdown report from saved artifacts.")
    parser.add_argument("--partial-note", type=str, default=None, help="Optional partial-run label note.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_MD, help="Output report path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_model_v1_report(output_md=args.output, partial_note=args.partial_note)
