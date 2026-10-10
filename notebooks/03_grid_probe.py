"""Boundary Ingestion, Spatial Gridding, and Feasibility Probe (Phase 3).

Downloads India state boundaries from DataMeet maps repository,
builds 0.5-degree spatial candidate grid, calculates equal-area inside shares (EPSG:6933),
evaluates cell inclusion rules, maps active fire detections, and exports grid artifacts.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BOUNDARIES_DIR = PROJECT_ROOT / "data" / "raw" / "boundaries"
RAW_FIRMS_DIR = PROJECT_ROOT / "data" / "raw" / "firms"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

DATAMEET_FILES = ["Admin2.shp", "Admin2.shx", "Admin2.dbf", "Admin2.prj", "Admin2.cpg"]
DATAMEET_BASE_URL = "https://raw.githubusercontent.com/datameet/maps/master/States/"


def run_grid_probe():
    print("=" * 80)
    print("PHASE 3: BOUNDARIES, SPATIAL GRIDDING & COVERAGE AUDIT")
    print("=" * 80)

    # 1. Boundary Verification
    shp_path = RAW_BOUNDARIES_DIR / "Admin2.shp"
    if not shp_path.exists():
        print(f"ERROR: {shp_path} not found. Please download from DataMeet repo.", file=sys.stderr)
        sys.exit(1)

    print(f"1. Loading boundaries from {shp_path}...")
    gdf = gpd.read_file(shp_path)
    crs_orig = str(gdf.crs)
    feature_count = len(gdf)
    print(f"CRS: {crs_orig} | Feature count (States & UTs): {feature_count}")

    # Geometry validity check and repair
    invalid_mask = ~gdf.is_valid
    invalid_count = invalid_mask.sum()
    print(f"Invalid geometries found: {invalid_count}")
    if invalid_count > 0:
        gdf["geometry"] = gdf["geometry"].apply(shapely.make_valid)

    # Dissolve to national boundary
    india_gdf_4326 = gdf.dissolve()
    india_poly_4326 = india_gdf_4326.geometry.iloc[0]

    # Reproject to equal-area projection EPSG:6933
    gdf_ea = gdf.to_crs("EPSG:6933")
    india_gdf_ea = gdf_ea.dissolve()
    india_poly_ea = india_gdf_ea.geometry.iloc[0]

    area_km2 = india_poly_ea.area / 1e6
    print(f"Total National Land Area (EPSG:6933): {area_km2:,.2f} km2")

    sanity_min, sanity_max = 2_800_000, 3_600_000
    area_sanity = sanity_min <= area_km2 <= sanity_max
    print(f"Area sanity check [2.8M - 3.6M km2]: {'PASS' if area_sanity else 'FAIL'} ({area_km2:,.2f} km2)")
    assert area_sanity, f"Area {area_km2} outside sanity range!"

    # 2. Candidate Grid Generation (0.5 degree)
    print("\n" + "-" * 80)
    print("2. BUILDING CANDIDATE SPATIAL GRID (0.5 degree, Lat [6,38], Lon [67,98])")
    print("-" * 80)

    candidate_rows = []
    for r in range(64):
        lat_min = 6.0 + r * 0.5
        lat_max = lat_min + 0.5
        lat_c = lat_min + 0.25
        for c in range(62):
            lon_min = 67.0 + c * 0.5
            lon_max = lon_min + 0.5
            lon_c = lon_min + 0.25
            cell_id = f"r{r:02d}c{c:02d}"
            geom = box(lon_min, lat_min, lon_max, lat_max)
            candidate_rows.append({
                "cell_id": cell_id,
                "row": r,
                "col": c,
                "lat_c": round(lat_c, 4),
                "lon_c": round(lon_c, 4),
                "lat_min": round(lat_min, 4),
                "lat_max": round(lat_max, 4),
                "lon_min": round(lon_min, 4),
                "lon_max": round(lon_max, 4),
                "geometry": geom,
            })

    cand_gdf = gpd.GeoDataFrame(candidate_rows, crs="EPSG:4326")
    total_candidates = len(cand_gdf)
    print(f"Total candidate grid cells: {total_candidates} (64 rows x 62 cols)")

    # Spot check: (30.25, 78.75) falls in row 48, col 23
    test_pt = Point(78.75, 30.25)
    matched_test = cand_gdf[cand_gdf.geometry.contains(test_pt)].iloc[0]
    print(f"Spot Check: Coordinate (30.25, 78.75) -> cell_id: {matched_test.cell_id}, row: {matched_test.row}, col: {matched_test.col}")
    assert matched_test.row == 48 and matched_test.col == 23, "Spot check failed!"

    # 3. Inclusion Rule Calculation
    print("\n" + "-" * 80)
    print("3. COMPUTING INSIDE SHARES (Equal-Area EPSG:6933)")
    print("-" * 80)

    cand_ea = cand_gdf.to_crs("EPSG:6933")
    cand_ea["cell_area"] = cand_ea.geometry.area

    # Calculate intersection area with India
    print("Intersecting candidate cells with national boundary polygon...")
    intersections = cand_ea.geometry.intersection(india_poly_ea)
    cand_gdf["inside_share"] = (intersections.area / cand_ea["cell_area"]).clip(0.0, 1.0)
    cand_ea["inside_share"] = cand_gdf["inside_share"]

    # Point-in-polygon for cell centers
    [Point(xy) for xy in zip(cand_gdf["lon_c"], cand_gdf["lat_c"])]
    cand_gdf["center_in_india"] = shapely.contains_xy(india_poly_4326, cand_gdf["lon_c"].values, cand_gdf["lat_c"].values)

    # Threshold Counts
    cnt_center = int(cand_gdf["center_in_india"].sum())
    cnt_ge_25 = int((cand_gdf["inside_share"] >= 0.25).sum())
    cnt_ge_50 = int((cand_gdf["inside_share"] >= 0.50).sum())
    cnt_ge_75 = int((cand_gdf["inside_share"] >= 0.75).sum())

    print("Candidate cell inclusion counts:")
    print(f"  - Centre point within India:     {cnt_center:>5d} cells")
    print(f"  - inside_share >= 0.25:          {cnt_ge_25:>5d} cells")
    print(f"  - inside_share >= 0.50 (DEFAULT):{cnt_ge_50:>5d} cells")
    print(f"  - inside_share >= 0.75:          {cnt_ge_75:>5d} cells")

    # Sample 10 boundary cells with shares between 0.25 and 0.75
    sample_boundary = cand_gdf[(cand_gdf["inside_share"] >= 0.25) & (cand_gdf["inside_share"] < 0.75)].head(10)
    print("\nSample 10 cells with inside_share in [0.25, 0.75):")
    for _, row in sample_boundary.iterrows():
        print(f"  Cell {row.cell_id} (row {row.row:02d}, col {row.col:02d}, lat {row.lat_c:.2f}, lon {row.lon_c:.2f}): inside_share = {row.inside_share:.4f}")

    # Island cells check: Andaman and Nicobar & Lakshadweep
    an_cells = cand_gdf[(cand_gdf["lat_c"] >= 6.5) & (cand_gdf["lat_c"] <= 14.0) & (cand_gdf["lon_c"] >= 92.0) & (cand_gdf["lon_c"] <= 94.5)]
    lak_cells = cand_gdf[(cand_gdf["lat_c"] >= 8.0) & (cand_gdf["lat_c"] <= 12.5) & (cand_gdf["lon_c"] >= 71.5) & (cand_gdf["lon_c"] <= 74.0)]

    an_kept = an_cells[an_cells["inside_share"] >= 0.50]
    lak_kept = lak_cells[lak_cells["inside_share"] >= 0.50]
    print("\nIsland Cells Status (inside_share >= 0.50 rule):")
    print(f"  - Andaman & Nicobar candidate cells: {len(an_cells)}, kept: {len(an_kept)} (Max share: {an_cells['inside_share'].max():.4f})")
    print(f"  - Lakshadweep candidate cells: {len(lak_cells)}, kept: {len(lak_kept)} (Max share: {lak_cells['inside_share'].max():.4f})")

    # 4. Coverage of Detections
    print("\n" + "-" * 80)
    print("4. DETECTION SPATIAL COVERAGE AUDIT")
    print("-" * 80)

    firms_csv = RAW_FIRMS_DIR / "fire_archive_M-C61_808434.csv"
    print(f"Loading filtered FIRMS detections from {firms_csv}...")
    df_raw = pd.read_csv(firms_csv)

    # Filter: dedup, conf >= 30, type == 0
    df_dedup = df_raw.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"], keep="first")
    df_filtered = df_dedup[(df_dedup["confidence"] >= 30) & (df_dedup["type"] == 0)].copy()
    n_filtered_dets = len(df_filtered)
    print(f"Total filtered input detections: {n_filtered_dets:,}")

    # Index arithmetic for cell assignment
    det_rows = np.floor((df_filtered["latitude"].values - 6.0) / 0.5).astype(int)
    det_cols = np.floor((df_filtered["longitude"].values - 67.0) / 0.5).astype(int)

    # Valid grid index mask
    valid_grid_mask = (det_rows >= 0) & (det_rows < 64) & (det_cols >= 0) & (det_cols < 62)
    df_filtered["cell_id"] = [f"r{r:02d}c{c:02d}" if v else "OUT_OF_GRID" for r, c, v in zip(det_rows, det_cols, valid_grid_mask)]

    # Point-in-polygon test with vectorized contains_xy
    det_lons = df_filtered["longitude"].values
    det_lats = df_filtered["latitude"].values
    in_india_mask = shapely.contains_xy(india_poly_4326, det_lons, det_lats)
    df_filtered["in_india"] = in_india_mask

    # Map kept cell IDs (inside_share >= 0.50)
    kept_cells_set = set(cand_gdf[cand_gdf["inside_share"] >= 0.50]["cell_id"])
    df_filtered["in_kept_cell"] = df_filtered["cell_id"].isin(kept_cells_set)

    # Categorize detections
    total_dets = len(df_filtered)
    in_india_count = int(in_india_mask.sum())
    outside_india_count = total_dets - in_india_count

    in_kept_cells_count = int(df_filtered["in_kept_cell"].sum())
    in_india_dropped_cells_count = int((in_india_mask & (~df_filtered["in_kept_cell"])).sum())

    share_in_india = in_india_count / total_dets
    share_in_kept = in_kept_cells_count / total_dets
    share_in_india_dropped = in_india_dropped_cells_count / in_india_count if in_india_count > 0 else 0
    share_outside_india = outside_india_count / total_dets

    print("Detection Distribution Summary:")
    print(f"  - Total Filtered Detections:             {total_dets:>8,d} (100.0%)")
    print(f"  - Inside India Polygon:                 {in_india_count:>8,d} ({share_in_india*100:5.2f}%)")
    print(f"  - Inside Kept Cells (inside_share>=0.5):{in_kept_cells_count:>8,d} ({share_in_kept*100:5.2f}%)")
    print(f"  - Inside India but in Dropped Cells:    {in_india_dropped_cells_count:>8,d} ({share_in_india_dropped*100:5.2f}% of in-India)")
    print(f"  - Outside India Boundary:               {outside_india_count:>8,d} ({share_outside_india*100:5.2f}%)")

    # Save coverage table
    coverage_records = [
        {"category": "Total Filtered Detections", "count": total_dets, "share_of_total": 1.0},
        {"category": "Inside India Polygon", "count": in_india_count, "share_of_total": round(share_in_india, 4)},
        {"category": "Inside Kept Cells (inside_share >= 0.5)", "count": in_kept_cells_count, "share_of_total": round(share_in_kept, 4)},
        {"category": "Inside India but in Dropped Cells", "count": in_india_dropped_cells_count, "share_of_in_india": round(share_in_india_dropped, 4)},
        {"category": "Outside India Boundary", "count": outside_india_count, "share_of_total": round(share_outside_india, 4)},
    ]
    pd.DataFrame(coverage_records).to_csv(DOCS_AUDIT_DIR / "grid_coverage.csv", index=False)
    print(f"Saved coverage summary to {DOCS_AUDIT_DIR / 'grid_coverage.csv'}")

    # 5. Kept Cells Dataset & State Assignment
    print("\n" + "-" * 80)
    print("5. STATE ATTRIBUTION AND EXPORT OF KEPT CELLS")
    print("-" * 80)

    kept_cand_ea = cand_ea[cand_ea["inside_share"] >= 0.50].copy()
    kept_count = len(kept_cand_ea)
    print(f"Total Kept Grid Cells: {kept_count}")

    # Assign primary state (largest area overlap)
    print("Calculating state overlaps for kept cells...")
    cell_states = []
    for _, cell_row in kept_cand_ea.iterrows():
        cell_geom = cell_row.geometry
        # Intersect with all states in gdf_ea
        overlaps = gdf_ea.geometry.intersection(cell_geom)
        overlap_areas = overlaps.area
        best_idx = overlap_areas.argmax()
        best_state = gdf_ea.iloc[best_idx]["ST_NM"]
        cell_states.append(best_state)

    kept_cand_ea["state"] = cell_states

    # Export grid_cells_0.5deg.parquet
    export_cols = [
        "cell_id", "row", "col", "lat_c", "lon_c",
        "lat_min", "lat_max", "lon_min", "lon_max",
        "inside_share", "state"
    ]
    grid_cells_df = pd.DataFrame(kept_cand_ea.drop(columns=["geometry", "cell_area"]))[export_cols]
    
    parquet_path = PROCESSED_DIR / "grid_cells_0.5deg.parquet"
    grid_cells_df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(grid_cells_df)} grid cells to {parquet_path}")

    # Export grid summary
    grid_summary = {
        "candidate_cells": total_candidates,
        "kept_cells_ge_50": kept_count,
        "cells_ge_25": cnt_ge_25,
        "cells_ge_75": cnt_ge_75,
        "national_area_km2": round(area_km2, 2),
        "est_cell_day_rows_study": kept_count * 4018,
        "est_weather_rows_spinup": kept_count * 4383,
        "share_in_india_dets_lost": round(share_in_india_dropped, 4),
    }
    pd.DataFrame([grid_summary]).to_csv(DOCS_AUDIT_DIR / "grid_summary.csv", index=False)
    print(f"Saved grid summary to {DOCS_AUDIT_DIR / 'grid_summary.csv'}")

    # 6. Panel Volume Estimates
    print("\n" + "-" * 80)
    print("6. PANEL MATRIX DATA VOLUME PROJECTIONS")
    print("-" * 80)
    print("Study Period Days (2015-2025): 4,018 days")
    print(f"Total ML Cell-Days (Kept Cells x 4,018): {kept_count * 4018:,} rows")
    print(f"Spin-Up Weather Rows (Kept Cells x 4,383): {kept_count * 4383:,} rows")

    # 7. Stop Gate 3 Evaluation
    print("\n" + "-" * 80)
    print("7. STOP GATE 3 EVALUATION")
    print("-" * 80)
    print(f"Kept Cell Count: {kept_count} (Target: 1,000 to 1,400) -> {'PASS' if 1000 <= kept_count <= 1400 else 'FAIL'}")
    print(f"Lost In-India Detection Share: {share_in_india_dropped * 100:.2f}% (Limit: < 10%) -> {'PASS' if share_in_india_dropped < 0.10 else 'FAIL'}")
    print(f"Boundary Area: {area_km2:,.2f} km2 (Target: 2.8M to 3.6M km2) -> {'PASS' if area_sanity else 'FAIL'}")

    assert 1000 <= kept_count <= 1400, f"Kept cell count {kept_count} outside 1,000-1,400 range"
    assert share_in_india_dropped < 0.10, f"Lost detection share {share_in_india_dropped:.4f} >= 10%"
    assert area_sanity, "Boundary area out of range"

    print("\nSTOP GATE 3 VERDICT: PASS [OK]")
    print("\nPhase 3 Execution Completed Successfully.")


if __name__ == "__main__":
    run_grid_probe()
