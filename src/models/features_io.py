"""Feature and label loading interface with chronological split guards.

Loads engineered feature tables and joins them strictly with split-verified
fire cell-day labels while enforcing test-split access protection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.models.dataset import load_split

FEATURES_FIRE_BASE_PATH = PROCESSED_DIR / "features_fire_base.parquet"
FEATURES_FULL_PATH = PROCESSED_DIR / "features.parquet"


def load_features(
    split: str,
    label: str,
    feature_names: list[str],
    primary: bool = True,
    features_path: Path | str | None = None,
    fire_table_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load features, target labels, and keys for a given split.

    Args:
        split: 'train' or 'val' (or 'test' only if ALLOW_TEST_SPLIT=1). 'gap' is refused.
        label: 'new_fire_next_48h' or 'fire_next_48h'.
        feature_names: Ordered list of feature columns to load.
        primary: If True and label == 'new_fire_next_48h', filters to ~burning_prev3.
        features_path: Optional path to features Parquet file. Defaults to FEATURES_FULL_PATH
            if it exists, else FEATURES_FIRE_BASE_PATH.
        fire_table_path: Optional path to fire_cell_day.parquet.

    Returns:
        Tuple of (X_features, y_label, keys_df) where keys_df contains
        ['cell_id', 'date', 'burning_prev3', 'split'].
    """
    # Step 1: Load label split via guarded load_split (refuses test/gap appropriately)
    split_df = load_split(split=split, label=label, primary=primary, fire_table_path=fire_table_path)

    # Step 2: Resolve features path
    if features_path is not None:
        f_path = Path(features_path)
    elif FEATURES_FULL_PATH.exists():
        f_path = FEATURES_FULL_PATH
    else:
        f_path = FEATURES_FIRE_BASE_PATH

    if not f_path.exists():
        raise FileNotFoundError(f"Features file not found at {f_path}")

    cols_to_read = ["cell_id", "date"] + list(feature_names)
    feat_df = pd.read_parquet(f_path, columns=cols_to_read)

    # Ensure matching types for fast exact merge
    split_df = split_df.copy()
    split_df["cell_id"] = split_df["cell_id"].astype(str)
    feat_df["cell_id"] = feat_df["cell_id"].astype(str)

    merged = split_df.merge(feat_df, on=["cell_id", "date"], how="inner", validate="one_to_one")
    if len(merged) != len(split_df):
        raise ValueError(
            f"Feature merge row count mismatch for split='{split}': "
            f"expected {len(split_df)}, got {len(merged)}"
        )

    X = merged[list(feature_names)].copy()
    y = merged[label].copy()
    keys = merged[["cell_id", "date", "burning_prev3", "split"]].copy()

    return X, y, keys

