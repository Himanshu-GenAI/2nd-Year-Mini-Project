from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluate import (
    base_rate,
    brier,
    plot_reliability,
    pr_auc,
    precision_at_top_frac,
    recall_at_top_frac,
    record_result,
    reliability_table,
)


def test_perfect_scores():
    # 10 rows per day, 1 positive each day (10% positive)
    # Perfect score: score == label
    df = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(10)] * 2,
            "date": ["2020-01-01"] * 10 + ["2020-01-02"] * 10,
            "label": [1] + [0] * 9 + [1] + [0] * 9,
            "score": [1.0] + [0.0] * 9 + [1.0] + [0.0] * 9,
        }
    )
    assert pr_auc(df, "label", "score") == 1.0
    rec, dates_used = recall_at_top_frac(df, "label", "score", frac=0.10)
    assert rec == 1.0
    assert dates_used == 2
    assert brier(df, "label", "score") == 0.0


def test_random_scores_pr_auc_near_base_rate():
    # 20,000 samples, 5% positive
    rng = np.random.default_rng(42)
    n = 20000
    y_true = (rng.random(n) < 0.05).astype(int)
    scores = rng.random(n)
    df = pd.DataFrame({"label": y_true, "score": scores})

    br = base_rate(df, "label")
    prauc = pr_auc(df, "label", "score")
    # PR-AUC should be within 0.02 of base rate
    assert abs(prauc - br) < 0.02


def test_hand_computed_recall_at_top_frac():
    # 2 dates, 5 cells each (10 rows total)
    # Day 1: cells c1..c5. Positives: c1, c2 (2 positives).
    # Scores: c1: 0.9, c2: 0.8, c3: 0.7, c4: 0.1, c5: 0.2
    # frac = 0.20 -> ceil(0.20 * 5) = 1 cell selected.
    # Top 1 selected for Day 1: c1 (positive). Recall = 1 / 2 = 0.50.
    # Day 2: cells c1..c5. Positives: c3 (1 positive).
    # Scores: c1: 0.9, c2: 0.8, c3: 0.7, c4: 0.1, c5: 0.2
    # Top 1 selected for Day 2: c1 (negative). Recall = 0 / 1 = 0.00.
    # Mean recall = (0.50 + 0.00) / 2 = 0.250.
    df = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4", "c5"] * 2,
            "date": ["2020-01-01"] * 5 + ["2020-01-02"] * 5,
            "label": [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            "score": [0.9, 0.8, 0.7, 0.1, 0.2, 0.9, 0.8, 0.7, 0.1, 0.2],
        }
    )
    rec, dates_used = recall_at_top_frac(df, "label", "score", frac=0.20)
    assert dates_used == 2
    assert rec == 0.250

    # Precision at top 20%: Day 1 pos=1/1=1.0; Day 2 pos=0/1=0.0 -> mean 0.50
    prec = precision_at_top_frac(df, "label", "score", frac=0.20)
    assert prec == 0.50


def test_date_with_no_positives_excluded():
    # Day 1: 1 positive. Day 2: 0 positives.
    df = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"] * 2,
            "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "label": [1, 0, 0, 0],
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    rec, dates_used = recall_at_top_frac(df, "label", "score", frac=0.50)
    assert dates_used == 1  # Only 2020-01-01 used
    assert rec == 1.0


def test_tie_breaking_stability():
    # Identical scores on two cells
    df = pd.DataFrame(
        {
            "cell_id": ["c2", "c1"],
            "date": ["2020-01-01", "2020-01-01"],
            "label": [0, 1],
            "score": [0.5, 0.5],
        }
    )
    # Sorting by [score desc, cell_id asc] must always pick c1 before c2
    rec1, _ = recall_at_top_frac(df, "label", "score", frac=0.50)
    rec2, _ = recall_at_top_frac(df, "label", "score", frac=0.50)
    assert rec1 == rec2 == 1.0


def test_reliability_table_and_plot(tmp_path: Path):
    df = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(100)],
            "date": ["2020-01-01"] * 100,
            "label": [1] * 20 + [0] * 80,
            "score": np.linspace(0.01, 0.99, 100),
        }
    )
    table = reliability_table(df, "label", "score", bins=10)
    assert len(table) == 10
    # Bin counts should be equal (10 each)
    assert (table["count"] == 10).all()

    plot_file = tmp_path / "reliability.png"
    plot_reliability(table, plot_file)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 1000


def test_record_result_appends(tmp_path: Path):
    test_csv = tmp_path / "results.csv"
    test_md = tmp_path / "results.md"
    res = {
        "model": "test_model",
        "split": "val",
        "label": "fire_next_48h",
        "set_name": "all",
        "rows": 1000,
        "positives": 50,
        "base_rate": 0.05,
        "pr_auc": 0.25,
        "pr_auc_over_base": 5.0,
        "recall_top10": 0.40,
        "precision_top5": 0.30,
        "brier": 0.04,
        "dates_used": 10,
        "eval_date": "2026-11-09",
    }
    record_result(res, csv_path=test_csv, md_path=test_md)
    record_result(res, csv_path=test_csv, md_path=test_md)

    assert test_csv.exists()
    df_res = pd.read_csv(test_csv)
    assert len(df_res) == 2
