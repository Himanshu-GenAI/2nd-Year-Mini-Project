"""Dataset loader with temporal split protection.

Provides strict access controls across chronological train/val/test splits,
enforcing primary evaluation filters and test-set lookahead locks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.config import FIRE_TABLE_PATH
from src.data.labels import primary_mask


def load_split(
    split: str,
    label: str,
    primary: bool = True,
    fire_table_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load a specific temporal partition from the fire cell-day table.

    Args:
        split: Partition to load ('train', 'val', 'test'). 'gap' is strictly disallowed.
        label: Target label column name ('fire_next_48h' or 'new_fire_next_48h').
        primary: If True and label is 'new_fire_next_48h', filters to primary_mask.
        fire_table_path: Optional custom path to fire cell-day parquet.

    Returns:
        Filtered partition DataFrame.
    """
    valid_splits = {"train", "val", "test"}
    if split == "gap":
        raise ValueError("Loading the 'gap' split is strictly prohibited.")
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}.")

    if split == "test" and os.environ.get("ALLOW_TEST_SPLIT") != "1":
        raise PermissionError(
            "Access to 'test' split is locked until Week 6 final evaluation. "
            "Set ALLOW_TEST_SPLIT=1 to override if explicitly authorized."
        )

    path = Path(fire_table_path or FIRE_TABLE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Fire table not found at {path}. Run Day 3 pipeline first.")

    # Optimized read: filter on split column
    df = pd.read_parquet(path, filters=[("split", "==", split)])

    if primary and label == "new_fire_next_48h":
        df = df[primary_mask(df)].copy()

    return df
