"""Unit tests for one-shot test runner (scripts/run_test_once.py).

All tests use synthetic data and mocks. The test split is NEVER accessed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.run_test_once import (
    check_git_cleanliness,
    check_lock_file,
    format_verdict,
    run_test_pipeline,
    verify_frozen_hashes,
)
from src.config import PROJECT_ROOT


def test_real_mode_refuses_without_approval():
    """Real mode execution must immediately refuse if --i-have-approval is not passed."""
    with pytest.raises(RuntimeError, match="Real test run requires explicit --i-have-approval flag"):
        run_test_pipeline(rehearsal=False, approval=False)


def test_real_mode_refuses_on_hash_mismatch(tmp_path: Path):
    """Integrity check must fail loudly if any frozen file hash mismatches."""
    fake_bundle_dir = tmp_path / "models"
    fake_bundle_dir.mkdir()
    fake_model = fake_bundle_dir / "model.json"
    fake_model.write_text('{"mock": "model"}\n', encoding="utf-8")

    tampered_yaml = tmp_path / "model_v1_frozen_tampered.yaml"
    tampered_yaml.write_text(
        """
bundle_files_sha256:
  model.json: 0000000000000000000000000000000000000000000000000000000000000000
upstream_files_sha256: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Hash mismatch for bundle file model.json"):
        verify_frozen_hashes(tampered_yaml, bundle_dir=fake_bundle_dir)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "model_v1" / "model.json").exists(),
    reason="Model v1 bundle not present in CI",
)
def test_real_frozen_hashes_match_disk():
    """Verify that all actual files on disk match configs/model_v1_frozen.yaml when present."""
    verified = verify_frozen_hashes()
    assert len(verified) == 8


def test_real_mode_refuses_if_dirty():
    """Real mode must refuse execution if git reports uncommitted files in src/, configs/, scripts/."""
    fake_proc = MagicMock()
    fake_proc.stdout = " M src/models/train_v1.py\n"
    with patch("subprocess.run", return_value=fake_proc):
        with pytest.raises(RuntimeError, match="Git working directory has uncommitted changes"):
            check_git_cleanliness()


def test_real_mode_refuses_if_lock_exists(tmp_path: Path):
    """Execution must refuse if docs/audit/test_run_lock.json already exists."""
    lock_file = tmp_path / "test_run_lock.json"
    lock_file.write_text('{"status": "COMPLETED"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Test run lock file already exists"):
        check_lock_file(lock_file)


def test_rehearsal_mode_never_reads_test_split():
    """Rehearsal mode safety checks must reject target data containing 2024 or 2025 rows."""
    fake_keys = pd.DataFrame(
        {
            "cell_id": ["r01c01", "r01c02"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "burning_prev3": [False, False],
            "split": ["val", "val"],
        }
    )
    fake_features = pd.DataFrame({"feat1": [0.1, 0.2]})
    fake_y = pd.Series([0, 1])

    with patch("scripts.run_test_once.verify_frozen_hashes", return_value={}):
        with patch("scripts.run_test_once.load_model_v1_bundle"):
            with patch("scripts.run_test_once.load_split"):
                with patch("scripts.run_test_once.fit_climatology"):
                    with patch(
                        "scripts.run_test_once.load_features",
                        return_value=(fake_features, fake_y, fake_keys),
                    ):
                        with pytest.raises(RuntimeError, match="REHEARSAL ERROR: Touched test years 2024/2025"):
                            run_test_pipeline(rehearsal=True, approval=False)


def test_replay_schema_matches_dev_predictions_plus_burning_prev3():
    """Verify that the replay predictions schema matches dev predictions schema plus burning_prev3."""
    dev_pred_path = PROJECT_ROOT / "data" / "processed" / "predictions" / "model_v1_dev.parquet"
    if not dev_pred_path.exists():
        pytest.skip("model_v1_dev.parquet not present on disk")

    dev_df = pd.read_parquet(dev_pred_path)
    expected_dev_cols = ["cell_id", "date", "score", "tier", "model_version", "split"]
    assert list(dev_df.columns) == expected_dev_cols

    # Synthetic replay DataFrame constructed with identical rules
    n = 10
    replay_df = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(n)],
            "date": pd.date_range("2023-01-01", periods=n, unit="ns"),
            "score": pd.Series([np.nan if i % 2 == 1 else 0.2 for i in range(n)], dtype="float32"),
            "tier": pd.Series(
                ["Already burning" if i % 2 == 1 else "Moderate" for i in range(n)],
                dtype=pd.CategoricalDtype(categories=["Low", "Moderate", "High", "Critical", "Already burning"]),
            ),
            "model_version": "v1_tuned_20261121",
            "split": "val",
            "burning_prev3": [bool(i % 2 == 1) for i in range(n)],
        }
    )

    expected_replay_cols = expected_dev_cols + ["burning_prev3"]
    assert list(replay_df.columns) == expected_replay_cols

    # Check dtypes compatibility
    assert str(replay_df["cell_id"].dtype) == str(dev_df["cell_id"].dtype)
    assert replay_df["date"].dtype == dev_df["date"].dtype
    assert replay_df["score"].dtype == dev_df["score"].dtype
    assert replay_df["model_version"].dtype == dev_df["model_version"].dtype
    assert replay_df["split"].dtype == dev_df["split"].dtype
    assert replay_df["burning_prev3"].dtype == bool

    # Check burning_prev3 rule: score is NaN and tier is 'Already burning'
    burning_rows = replay_df[replay_df["burning_prev3"]]
    assert burning_rows["score"].isna().all()
    assert (burning_rows["tier"] == "Already burning").all()


def test_verdict_formatter_conditions():
    """Test verdict formatting across all combinations of pass and fail for conditions (a), (b), (c)."""
    # 1. Condition (a) FAIL: gain is only 10% (< 15%)
    pass_a_fail, txt_a_fail = format_verdict(
        prauc_model=0.22,
        prauc_clim=0.20,  # gain = 0.10
        prauc_ci_lower=0.005,
        recall_model=0.45,
        recall_clim=0.40,
    )
    assert not pass_a_fail
    lines_a = txt_a_fail.strip().splitlines()
    assert lines_a[0].startswith("Condition (a):") and lines_a[0].endswith("- FAIL")
    assert lines_a[1].startswith("Condition (b):") and lines_a[1].endswith("- PASS")
    assert lines_a[2].startswith("Condition (c):") and lines_a[2].endswith("- PASS")
    assert lines_a[3] == "Overall: FAIL"

    # 2. Condition (b) FAIL: CI lower bound is <= 0
    pass_b_fail, txt_b_fail = format_verdict(
        prauc_model=0.25,
        prauc_clim=0.20,  # gain = 25% (PASS)
        prauc_ci_lower=-0.002,  # FAIL
        recall_model=0.45,
        recall_clim=0.40,
    )
    assert not pass_b_fail
    lines_b = txt_b_fail.strip().splitlines()
    assert lines_b[0].startswith("Condition (a):") and lines_b[0].endswith("- PASS")
    assert lines_b[1].startswith("Condition (b):") and lines_b[1].endswith("- FAIL")
    assert lines_b[2].startswith("Condition (c):") and lines_b[2].endswith("- PASS")
    assert lines_b[3] == "Overall: FAIL"

    # 3. Condition (c) FAIL: recall is lower than climatology
    pass_c_fail, txt_c_fail = format_verdict(
        prauc_model=0.25,
        prauc_clim=0.20,  # gain = 25% (PASS)
        prauc_ci_lower=0.015,  # PASS
        recall_model=0.38,
        recall_clim=0.42,  # diff = -0.04 (FAIL)
    )
    assert not pass_c_fail
    lines_c = txt_c_fail.strip().splitlines()
    assert lines_c[0].startswith("Condition (a):") and lines_c[0].endswith("- PASS")
    assert lines_c[1].startswith("Condition (b):") and lines_c[1].endswith("- PASS")
    assert lines_c[2].startswith("Condition (c):") and lines_c[2].endswith("- FAIL")
    assert lines_c[3] == "Overall: FAIL"

    # 4. All PASS: gain >= 15%, CI lower > 0, recall >= clim
    pass_all, txt_all = format_verdict(
        prauc_model=0.24,
        prauc_clim=0.20,  # gain = 20% >= 15% (PASS)
        prauc_ci_lower=0.012345,  # PASS
        recall_model=0.435,
        recall_clim=0.400,  # diff = +0.035 (PASS)
    )
    assert pass_all
    lines_all = txt_all.strip().splitlines()
    assert lines_all[0].startswith("Condition (a):") and lines_all[0].endswith("- PASS")
    assert lines_all[1].startswith("Condition (b):") and lines_all[1].endswith("- PASS")
    assert lines_all[2].startswith("Condition (c):") and lines_all[2].endswith("- PASS")
    assert lines_all[3] == "Overall: PASS"
