"""Evaluation harness for wildfire hazard forecasting models.

Computes precision-recall metrics, operational top-k capacity recalls,
Brier calibration scores, and reliability diagrams.
"""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from src.config import PROJECT_ROOT
from src.data.labels import primary_mask
from src.utils.plotting import plt

RESULTS_CSV_PATH = PROJECT_ROOT / "docs" / "results.csv"
RESULTS_MD_PATH = PROJECT_ROOT / "docs" / "results.md"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def base_rate(df: pd.DataFrame, label: str) -> float:
    """Calculate empirical positive prevalence."""
    if len(df) == 0:
        return 0.0
    return float(df[label].mean())


def pr_auc(df: pd.DataFrame, label: str, score: str) -> float:
    """Calculate Area Under the Precision-Recall Curve (Average Precision)."""
    y_true = df[label].values
    if int(y_true.sum()) == 0:
        return float("nan")
    y_score = df[score].values
    return float(average_precision_score(y_true, y_score))


def recall_at_top_frac(
    df: pd.DataFrame,
    label: str,
    score: str,
    frac: float = 0.10,
    cell_col: str = "cell_id",
    date_col: str = "date",
) -> tuple[float, int]:
    """Calculate mean daily recall at top fraction of predicted grid cells.

    Ties are stably broken by sorting ascending on cell_col.
    Dates with 0 positive fires are excluded from the average.
    """
    recalls = []

    # Group by date
    for _, day_df in df.groupby(date_col):
        pos_total = int(day_df[label].sum())
        if pos_total == 0:
            continue

        n_day = len(day_df)
        k = math.ceil(frac * n_day)

        # Stable sort: score descending, cell_id ascending
        sorted_day = day_df.sort_values(by=[score, cell_col], ascending=[False, True])
        top_k = sorted_day.iloc[:k]
        pos_in_top_k = int(top_k[label].sum())

        recall = pos_in_top_k / pos_total
        recalls.append(recall)

    dates_used = len(recalls)
    mean_recall = float(np.mean(recalls)) if dates_used > 0 else 0.0
    return mean_recall, dates_used


def precision_at_top_frac(
    df: pd.DataFrame,
    label: str,
    score: str,
    frac: float = 0.05,
    cell_col: str = "cell_id",
    date_col: str = "date",
) -> float:
    """Calculate mean daily precision at top fraction of predicted cells."""
    precisions = []

    for _, day_df in df.groupby(date_col):
        pos_total = int(day_df[label].sum())
        if pos_total == 0:
            continue

        n_day = len(day_df)
        k = math.ceil(frac * n_day)

        sorted_day = day_df.sort_values(by=[score, cell_col], ascending=[False, True])
        top_k = sorted_day.iloc[:k]
        pos_in_top_k = int(top_k[label].sum())

        precision = pos_in_top_k / k
        precisions.append(precision)

    return float(np.mean(precisions)) if precisions else 0.0


def brier(df: pd.DataFrame, label: str, score: str) -> float:
    """Calculate Brier calibration score (mean squared error of probabilities)."""
    y_true = df[label].values
    y_score = df[score].values
    return float(brier_score_loss(y_true, y_score))


def reliability_table(
    df: pd.DataFrame,
    label: str,
    score: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Generate equal-count quantile calibration table."""
    scores = df[score].values
    labels = df[label].values

    # Equal-count quantile bins via percentiles
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(scores, quantiles)
    # Ensure distinct edges
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    # Assign bins using searchsorted / digitize
    bin_indices = np.digitize(scores, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    table_rows = []
    for b in range(bins):
        mask = bin_indices == b
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(scores[mask].mean())
            obs_rate = float(labels[mask].mean())
        else:
            mean_pred = float("nan")
            obs_rate = float("nan")

        table_rows.append(
            {
                "bin": b,
                "mean_pred": mean_pred,
                "observed_rate": obs_rate,
                "count": count,
            }
        )

    return pd.DataFrame(table_rows)


def plot_reliability(table: pd.DataFrame, path: Path | str) -> None:
    """Plot reliability curve with 45-degree reference line and save figure."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    valid = table.dropna(subset=["mean_pred", "observed_rate"])

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(valid["mean_pred"], valid["observed_rate"], "s-", color="#1f77b4", label="Model Reliability")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate(
    df: pd.DataFrame,
    label: str,
    score: str,
    model: str,
    split: str,
    set_name: str = "primary",
) -> dict[str, Any]:
    """Execute comprehensive model evaluation across precision, recall, and calibration."""
    eval_df = df.copy()
    if label == "new_fire_next_48h" and set_name == "primary":
        eval_df = eval_df[primary_mask(eval_df)]

    n_rows = len(eval_df)
    n_pos = int(eval_df[label].sum())
    base_r = base_rate(eval_df, label)
    prauc = pr_auc(eval_df, label, score)
    prauc_over_base = prauc / base_r if (base_r > 0 and not math.isnan(prauc)) else float("nan")
    rec_top10, dates_used = recall_at_top_frac(eval_df, label, score, frac=0.10)
    prec_top5 = precision_at_top_frac(eval_df, label, score, frac=0.05)
    brier_score = brier(eval_df, label, score)

    result = {
        "model": model,
        "split": split,
        "label": label,
        "set_name": set_name,
        "rows": n_rows,
        "positives": n_pos,
        "base_rate": round(base_r, 6),
        "pr_auc": round(prauc, 6),
        "pr_auc_over_base": round(prauc_over_base, 3) if not math.isnan(prauc_over_base) else float("nan"),
        "recall_top10": round(rec_top10, 4),
        "precision_top5": round(prec_top5, 4),
        "brier": round(brier_score, 6),
        "dates_used": dates_used,
        "eval_date": str(date.today()),
    }
    return result


def record_result(
    result: dict[str, Any],
    csv_path: Path | str | None = None,
    md_path: Path | str | None = None,
) -> None:
    """Append evaluation metric dictionary to CSV and Markdown tables."""
    c_path = Path(csv_path or RESULTS_CSV_PATH)
    m_path = Path(md_path or RESULTS_MD_PATH)

    c_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "split",
        "label",
        "set_name",
        "rows",
        "positives",
        "base_rate",
        "pr_auc",
        "pr_auc_over_base",
        "recall_top10",
        "precision_top5",
        "brier",
        "dates_used",
        "eval_date",
    ]

    write_csv_header = not c_path.exists()
    with open(c_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_csv_header:
            writer.writeheader()
        writer.writerow(result)

    # Markdown Table append
    if not m_path.exists():
        with open(m_path, "w", encoding="utf-8") as f:
            f.write("# Model Benchmark Evaluation Results\n\n")
            f.write("| Model | Split | Label | Set | Rows | Positives | Base Rate | PR-AUC | PR/Base | Recall@10% | Prec@5% | Brier | Eval Date |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    md_row = (
        f"| {result['model']} | {result['split']} | {result['label']} | {result['set_name']} | "
        f"{result['rows']:,} | {result['positives']:,} | {result['base_rate']:.4f} | {result['pr_auc']:.4f} | "
        f"{result['pr_auc_over_base']:.2f} | {result['recall_top10']:.4f} | {result['precision_top5']:.4f} | "
        f"{result['brier']:.6f} | {result['eval_date']} |\n"
    )
    with open(m_path, "a", encoding="utf-8") as f:
        f.write(md_row)
