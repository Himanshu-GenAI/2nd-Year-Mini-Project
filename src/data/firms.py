"""NASA FIRMS MODIS Active Fire Telemetry Ingestion and Grid Assignment.

Loads raw FIRMS satellite observations, applies deduplication, filters
vegetation fire types and detection confidence thresholds, maps coordinates
to 0.5-degree grid cells, and exports clean partitioned telemetry.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    CONF_MIN,
    DETECTIONS_CLEAN_PATH,
    FIRE_TYPE_KEEP,
    FIRMS_CSV_PATH,
    PROCESSED_AUDIT_DIR,
    PROJECT_ROOT,
)
from src.data.grid import cell_row_col, load_grid, make_cell_id

logger = logging.getLogger(__name__)

DOCS_AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"


def load_raw(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load raw FIRMS CSV telemetry with strict dtypes and validation."""
    path = Path(csv_path or FIRMS_CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(f"FIRMS archive not found at {path}")

    cols = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "confidence",
        "type",
        "satellite",
        "frp",
    ]
    dtypes = {
        "latitude": "float64",
        "longitude": "float64",
        "acq_time": "int32",
        "confidence": "int32",
        "type": "int32",
        "satellite": "string",
        "frp": "float32",
    }

    df = pd.read_csv(
        path,
        usecols=cols,
        dtype=dtypes,
        parse_dates=["acq_date"],
    )
    # Ensure date component only
    df["acq_date"] = df["acq_date"].dt.date
    row_count = len(df)
    print(f"Loaded raw FIRMS rows: {row_count:,}")
    return df


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop duplicate telemetry on (latitude, longitude, acq_date, acq_time, satellite)."""
    n_before = len(df)
    subset = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
    cleaned = df.drop_duplicates(subset=subset, keep="first").copy()
    n_removed = n_before - len(cleaned)
    return cleaned, n_removed


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter records by fire type (presumed vegetation) and confidence."""
    n_before = len(df)

    # 1. Type filter
    type_mask = df["type"] == FIRE_TYPE_KEEP
    n_type_dropped = int((~type_mask).sum())
    filtered_type = df[type_mask]

    # 2. Confidence filter
    conf_mask = filtered_type["confidence"] >= CONF_MIN
    n_conf_dropped = int((~conf_mask).sum())
    final_df = filtered_type[conf_mask].copy()

    counts = {
        "type_dropped": n_type_dropped,
        "conf_dropped": n_conf_dropped,
        "total_filtered_dropped": n_before - len(final_df),
    }
    return final_df, counts


def assign_cells(df: pd.DataFrame, grid_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """Map telemetry coordinates to spatial grid and filter to kept cells."""
    if grid_df is None:
        grid_df = load_grid()

    kept_cell_set = set(grid_df["cell_id"])

    rows, cols = cell_row_col(df["latitude"].values, df["longitude"].values)
    cell_ids = make_cell_id(rows, cols)

    df_assigned = df.copy()
    df_assigned["cell_id"] = cell_ids

    # Keep rows whose cell_id is within kept cells
    in_kept_mask = df_assigned["cell_id"].isin(kept_cell_set)
    n_outside = int((~in_kept_mask).sum())

    df_kept = df_assigned[in_kept_mask].copy()
    return df_kept, n_outside


def build_clean_detections(
    csv_path: Path | str | None = None,
    output_path: Path | str | None = None,
    write_docs_audit: bool = True,
) -> pd.DataFrame:
    """Execute end-to-end FIRMS cleaning and cell assignment pipeline.

    Order of operations:
      1. Load raw telemetry
      2. Drop duplicates
      3. Apply filters (type == FIRE_TYPE_KEEP, confidence >= CONF_MIN)
      4. Assign grid cells and retain detections in kept cells
      5. Save parquet and record audit trail
    """
    is_custom_run = csv_path is not None or output_path is not None
    out_file = Path(output_path or DETECTIONS_CLEAN_PATH)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    grid_df = load_grid()

    # Step 1: Raw load
    df_raw = load_raw(csv_path)
    n_raw = len(df_raw)

    # Step 2: Duplicates
    df_dedup, n_dups = drop_duplicates(df_raw)
    n_after_dedup = len(df_dedup)

    # Step 3: Filters
    df_filtered, filter_counts = apply_filters(df_dedup)
    n_after_filters = len(df_filtered)

    # Step 4: Cell assignment
    df_clean, n_outside_cells = assign_cells(df_filtered, grid_df)
    n_final = len(df_clean)

    # Reorder columns
    clean_cols = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "confidence",
        "frp",
        "cell_id",
    ]
    df_clean = df_clean[clean_cols].reset_index(drop=True)

    # Save clean detections
    df_clean.to_parquet(out_file, index=False)
    print(f"Clean detections saved to {out_file} ({n_final:,} rows)")

    # Step 5: Save audit trail
    audit_rows = [
        {"step": "raw", "count": n_raw, "dropped": 0},
        {"step": "after_deduplication", "count": n_after_dedup, "dropped": n_dups},
        {"step": "after_filters", "count": n_after_filters, "dropped": filter_counts["total_filtered_dropped"]},
        {"step": "after_cell_assignment", "count": n_final, "dropped": n_outside_cells},
    ]
    audit_df = pd.DataFrame(audit_rows)

    if not is_custom_run or write_docs_audit:
        audit_path_processed = PROCESSED_AUDIT_DIR / "firms_load_counts.csv"
        audit_path_docs = DOCS_AUDIT_DIR / "firms_load_counts.csv"
        audit_df.to_csv(audit_path_processed, index=False)
        audit_df.to_csv(audit_path_docs, index=False)
        print(f"Audit counts written to {audit_path_processed} and {audit_path_docs}")

    return df_clean


def load_clean_detections(path: Path | str | None = None) -> pd.DataFrame:
    """Load previously saved clean detections Parquet."""
    clean_path = Path(path or DETECTIONS_CLEAN_PATH)
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean detections file not found at {clean_path}. Run build_clean_detections() first.")
    return pd.read_parquet(clean_path)
