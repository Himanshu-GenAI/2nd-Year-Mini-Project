@echo off
REM Open-Meteo Weather Harvester Runner for Windows Task Scheduler
REM Direct execution using virtual environment interpreter

set PROJECT_DIR=%~dp0..
cd /d "%PROJECT_DIR%"

"%PROJECT_DIR%\.venv\Scripts\python.exe" "%PROJECT_DIR%\scripts\run_weather_fetch.py" --max-minutes 125
