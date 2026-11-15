# Wildfire Early Alert Dashboard (Workspace: `ashutosh/`)

Interactive Streamlit + Folium dashboard module for the Wildfire Early Alert project (PS-113, India).  
This module is isolated inside `ashutosh/` and does not alter the root project or existing models.

---

## 1. Quickstart (Windows / PowerShell)

All commands are run from inside the `ashutosh/` directory.

### Step 1: Set Up Virtual Environment & Dependencies
```powershell
cd ashutosh
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 2: Run Tests & Linter
```powershell
# Run the complete test suite (uses ashutosh/pytest.ini)
.\.venv\Scripts\python.exe -m pytest -v

# Run the linter (uses ashutosh/ruff.toml)
.\.venv\Scripts\ruff.exe check src tests
```

### Step 3: Launch the Dashboard
```powershell
# Launch in default mock mode (reads local 2023 Parquet files)
.\.venv\Scripts\streamlit.exe run src\dashboard\app.py
```
Open `http://localhost:8501` in your browser.

---

## 2. Pinned Dependencies

The following direct dependencies are pinned in `ashutosh/requirements.txt`:
```
streamlit==1.67.0
streamlit-folium==0.27.4
folium==0.20.0
requests==2.34.2
pandas==3.1.1
pyarrow==26.0.1
pyyaml==6.0.3
pytest==9.2.0
ruff==0.17.5
```

---

## 3. Configuration & Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `DASHBOARD_MODE` | `mock` | Switch between `mock` (local datasets) and `api` (Dhruv's REST endpoints). |
| `API_BASE_URL` | `http://localhost:8000` | Target URL for backend endpoints when `DASHBOARD_MODE=api`. |
| `DATA_DIR` | `../../data` | Root data directory containing `processed/grid_cells_0.5deg.parquet`, `predictions/`, `shap/`. |
| `TIERS_CONFIG` | `../../configs/tiers_v0_preview.yaml` | Path to preview tier percentile cutoffs. |

To run against a live local API:
```powershell
$env:DASHBOARD_MODE="api"
$env:API_BASE_URL="http://localhost:8000"
.\.venv\Scripts\streamlit.exe run src\dashboard\app.py
```

---

## 4. Integration Requirements with Main Repo & Dhruv's API

### A. API Endpoints Contract
When operating in `DASHBOARD_MODE=api`, the dashboard queries the following endpoints:
- `GET /api/health` -> `{status: "ok", model_version: str, date_min: "YYYY-MM-DD", date_max: "YYYY-MM-DD", n_cells: int}`
- `GET /api/regions` -> list of `{state: str, n_cells: int}`, sorted by state name with `"All India"` first
- `GET /api/risk?region=<state>&date=YYYY-MM-DD` -> GeoJSON `FeatureCollection` where each feature is a cell polygon with properties `cell_id, state, score, tier, date, model_version`, and top-level meta `{total_cells: int, scored_cells: int}`
- `GET /api/explain/{cell_id}?date=YYYY-MM-DD` -> `{cell_id: str, date: str, score: float, tier: str, drivers: list[{rank: int, driver: str, shap: float, feature_detail: str}]}`
- `POST /api/report` -> Not implemented in dashboard (returns 501 in backend)

### B. Tier Ranges Configuration Handling
- Legend score ranges (e.g. `< 13.6%`, `13.6% – 27.4%`, `27.4% – 46.3%`, `≥ 46.3%`) are dynamically derived from `TIERS_CONFIG`.
- **API Mode Fallback:** If `TIERS_CONFIG` is not available or absent on the host in `api` mode, the dashboard gracefully renders tier names (`Low`, `Moderate`, `High`, `Critical`) without numeric score ranges.
- **Initial Date Selection Fallback:** In `mock` mode, the dashboard selects the date with the highest count of High and Critical cells. In `api` mode, the dashboard uses `date_max` from `/api/health` as a fallback since the API contract does not provide global distribution statistics.

---

## 5. Known Limits

1. **Temporal Horizon:** The development files contain 2023 validation data only (`2023-01-01` to `2023-12-29`). Dates outside 2023 are rejected with an explicit warning.
2. **Test Split Protection:** In accordance with the competition protocol, 2024–2025 test split data is never accessed or served.
3. **Primary Set Filtering:** In Model v0 preview datasets, only primary cell-days (`burning_prev3 == False`) are present. Cells actively burning on the forecast day are omitted from scoring in this preview file.
