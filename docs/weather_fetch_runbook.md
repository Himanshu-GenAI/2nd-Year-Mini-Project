# Open-Meteo Weather Harvester Operations Runbook

**Project:** Wildfire Early Alert System for India (PS-113)  
**Target:** 1,167 spatial cells across 4,383 days (2014-01-01 to 2025-12-31)  
**Variables:** Daily maximum 2m air temperature (`temperature_2m_max`) and precipitation sum (`precipitation_sum`)

---

## 1. Quick Start & Execution

### Starting a Harvest Session
To launch a standard background harvest session (default 125 minutes):
```cmd
scripts\run_weather_fetch.bat
```
Or directly from PowerShell / terminal:
```powershell
.venv\Scripts\python.exe scripts\run_weather_fetch.py --max-minutes 125
```

### Checking Progress
To inspect cached cells, quota consumption, and harvest ETA:
```powershell
.venv\Scripts\python.exe scripts\run_weather_fetch.py --status
```

### Dry Run Preview
To preview remaining cells and projected API weight without making requests:
```powershell
.venv\Scripts\python.exe scripts\run_weather_fetch.py --dry-run
```

---

## 2. Process Architecture & Safety Controls

### Resumability
- Cells are saved per-cell to Parquet: `data/raw/weather/cells/{cell_id}.parquet`.
- Before fetching, the harvester checks whether the cell Parquet exists and contains exactly 4,383 continuous valid rows.
- If a file is complete, it is automatically skipped. If corrupted or incomplete, it is deleted and re-fetched.

### Mutual Exclusion (Locking)
- A process lock is maintained at `data/raw/weather/fetch.lock` containing the active PID.
- If another active process holds the lock, new invocations immediately exit with code 4.
- If a previous process crashed or died, the lock is automatically reclaimed.

### Pacing & Quota Caps
- Safety caps are enforced via the rolling ledger at `data/raw/weather/quota_ledger.json`:
  - **1-Minute Cap:** 480 weight units
  - **1-Hour Cap:** 4,000 weight units
  - **24-Hour Cap:** 8,000 weight units
  - **Minimum Interval:** 55 seconds between request starts
- The fetcher paces requests at ~63 cells/hour (127 cells/day) over ~9.2 days total.

---

## 3. Exit Codes

| Exit Code | Meaning | Action Required |
|---|---|---|
| **0** | Clean completion (session time elapsed or all cells cached) | Normal. Run next scheduled session. |
| **2** | Rate limited (HTTP 429) | Halts session. Wait until the next UTC day or stated reset window before resuming. |
| **3** | Unrecoverable HTTP / network error | Check `logs/weather_fetch.log` and `data/raw/weather/weather_failed.csv`. |
| **4** | Lock file held by live process | Another instance is already running. Do not run concurrent fetchers. |

---

## 4. Handling HTTP 429 (Rate Limits)

- **Observed Behavior:** If Open-Meteo returns HTTP 429, the response body contains the reason.
- **Short-term throttling:** The fetcher waits 60s and retries up to 3 times before terminating.
- **Daily limit reached:** If the message mentions the daily cap, the fetcher immediately halts with exit code 2.
- **Resolution:** Open-Meteo daily limits reset at 00:00 UTC. Resume execution after 00:05 UTC.

---

## 5. Failure Recovery & Cache Backup

### Restoring a Failed Cell
Failed requests are logged to `data/raw/weather/weather_failed.csv`. To retry failed cells:
1. Review the failure reason in `data/raw/weather/weather_failed.csv`.
2. Delete the offending entry from `data/raw/weather/weather_failed.csv` (or clear the CSV).
3. Re-run the harvester; it will automatically attempt un-cached cells.

### Backing Up the Weather Cache
To snapshot all downloaded weather Parquets:
```powershell
Compress-Archive -Path data\raw\weather\cells\*.parquet -DestinationPath data\weather_backup_$(Get-Date -Format 'yyyyMMdd').zip
```

---

## 6. Daily Windows Task Registration (Human-Only)

To register a daily scheduled task in Windows Task Scheduler running at 01:00 UTC daily:
```cmd
schtasks /create /tn "Wildfire_Weather_Fetch" /tr "%CD%\scripts\run_weather_fetch.bat" /sc daily /st 06:30 /ri 120 /du 04:00
```
