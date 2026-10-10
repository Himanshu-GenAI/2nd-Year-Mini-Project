# 🔥 Wildfire Risk Early Warning & Forecasting System
*2nd-Year Mini Project*

An operational machine learning system for wildfire hazard risk forecasting, combining NASA FIRMS active fire satellite archives, ERA5 surface meteorological observations, and the recursive Keetch-Byram Drought Index (KBDI).

---

## 🏛️ Project Architecture

```text
2nd-Year-Mini-Project/
├── data/
│   ├── raw/          # firms/, weather/  (git-ignored)
│   └── processed/    # parquet, partitioned by year (git-ignored)
├── src/
│   ├── config.py         # grid size, paths, split dates, thresholds
│   ├── data/
│   │   ├── firms_ingest.py
│   │   ├── grid.py
│   │   ├── weather_fetch.py
│   │   └── kbdi.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── labels.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── train.py
│   │   └── calibrate.py
│   ├── explain/shap_utils.py
│   ├── api/main.py       # FastAPI
│   ├── reports/llm_report.py
│   └── dashboard/app.py  # Streamlit
├── tests/                # pytest — incl. leakage checks
├── notebooks/            # exploration only, no production logic
├── docs/decisions.md
├── requirements.txt      # pinned versions
└── .github/workflows/ci.yml
```

---

## ⚙️ Core Components

1. **Spatial Mesh (`src/data/grid.py`):**
   - Discretizes geographical coordinates into regular $0.25^\circ$ spatial grid cells (~$27.5\text{ km}$).
   - Handles coordinate-to-grid mapping, centroid extraction, and Moore-neighborhood adjacency lookups.

2. **Data Ingestion (`src/data/`):**
   - **`firms_ingest.py`**: Ingests NASA MODIS/VIIRS thermal anomaly archives, filters low-confidence detections, and aggregates daily cell-level fire counts, max FRP, and day/night splits.
   - **`weather_fetch.py`**: Pulls historical daily weather reanalysis (temperature, precipitation, relative humidity, wind speed) from Open-Meteo with local caching and offline simulation fallback.
   - **`kbdi.py`**: Calculates the Keetch-Byram Drought Index ($0$ to $800$), modeling cumulative upper-soil duff moisture deficit with rain interception thresholding ($5.08\text{ mm}$).

3. **Feature Engineering & Labels (`src/features/`):**
   - **`labels.py`**: Generates strictly forward-shifted target labels ($t+1, t+3, t+7$) with lookahead protection.
   - **`build_features.py`**: Creates cyclical calendar encodings, rolling weather metrics, backward-looking fire lags ($1\text{d}, 3\text{d}, 7\text{d}$), and spatial neighbor fire propagation indicators. Parquet files are saved partitioned by year (`year=YYYY`).

4. **Modeling & Probability Calibration (`src/models/`):**
   - **`baselines.py`**: Persistence baseline, climatological frequency baseline, and logistic regression benchmarks.
   - **`train.py`**: Chronological train/validation/test pipeline with class-imbalance-aware XGBoost.
   - **`calibrate.py`**: Post-hoc probability calibration (Isotonic Regression / Platt Scaling) to minimize Expected Calibration Error (ECE).

5. **Explainability & Reports (`src/explain/` & `src/reports/`):**
   - **`shap_utils.py`**: TreeSHAP feature attributions decomposing risk into meteorological, drought, and historical drivers.
   - **`llm_report.py`**: Automated executive wildfire intelligence briefings for incident commanders and forestry managers.

6. **Serving & Dashboard (`src/api/` & `src/dashboard/`):**
   - **`api/main.py`**: High-performance FastAPI microservice exposing `/predict`, `/predict/batch`, `/explain`, and `/health`.
   - **`dashboard/app.py`**: Streamlit interactive GIS dashboard with heatmaps, cell diagnostics, SHAP waterfall charts, and briefing generation.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Himanshu-GenAI/2nd-Year-Mini-Project.git
cd 2nd-Year-Mini-Project
pip install -r requirements.txt
```

### 2. Run Tests (Including Leakage Verification)
```bash
pytest tests/ -v
```

### 3. Launch the FastAPI Microservice
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is accessible at: `http://localhost:8000/docs`

### 4. Launch the Streamlit Monitoring Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open your browser at: `http://localhost:8501`

---

## 🛡️ Anti-Leakage Guarantee

Time-series geospatial ML is highly prone to lookahead bias and spatial autocorrelation leakage. This project enforces:
- **Chronological Separation:**
  $$\max(\text{Train Dates}) < \min(\text{Val Dates}) < \max(\text{Val Dates}) < \min(\text{Test Dates})$$
- **Strict Backward Lagging:** All lag features use $t-k$ ($k \ge 1$). Observations from day $t$ are never included in features for day $t$.
- **Automated CI Validation:** `tests/test_leakage.py` automatically checks and enforces these invariants on every push and pull request.
