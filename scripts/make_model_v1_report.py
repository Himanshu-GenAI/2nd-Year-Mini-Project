"""Generate docs/model_v1_report.md strictly from saved evaluation artifacts.

Raises FileNotFoundError if any required upstream artifact is missing so no
invented or placeholder metrics can ever be written.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.config import DOCS_AUDIT_DIR, DOCS_DIR, PROJECT_ROOT
from src.models.train_v1 import ABLATION_CSV_PATH, FEATURE_IMPORTANCE_CSV_PATH, assert_test_split_locked

DEFAULT_RESULTS_CSV = DOCS_DIR / "results.csv"
DEFAULT_BREAKDOWN_CSV = DOCS_AUDIT_DIR / "model_v1_breakdown.csv"
DEFAULT_TIERS_YAML = PROJECT_ROOT / "configs" / "tiers_preview.yaml"
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
    output_md: Path = DEFAULT_REPORT_MD,
    partial_note: str | None = None,
) -> str:
    """Build markdown report strictly from saved CSV/YAML files."""
    assert_test_split_locked()

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

    lines.extend(["## 1. Benchmark Comparison (`docs/results.csv`)", "", _df_to_markdown(res_df)])

    lines.extend(["", "## 2. Feature Group Ablation (`docs/audit/ablation.csv`)", "", _df_to_markdown(abl_df)])

    lines.extend(["", "## 3. Top Features (`docs/audit/feature_importance.csv`)", "", _df_to_markdown(imp_df.head(10))])

    lines.extend(["", "## 4. Slice Breakdown (`docs/audit/model_v1_breakdown.csv`)", "", _df_to_markdown(brk_df)])

    lines.extend(["", "## 5. Tier Preview Thresholds (`configs/tiers_preview.yaml`)", ""])
    lines.append(f"- **Model version:** `{tiers_cfg.get('model_version')}`")
    lines.append(f"- **Date:** `{tiers_cfg.get('date')}`")
    lines.append(f"- **Set:** `{tiers_cfg.get('set_name')}`")
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
