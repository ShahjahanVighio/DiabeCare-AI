@echo off
title DiabeCare AI - Local Server
cd /d "%~dp0"

echo ===================================================
echo Starting DiabeCare AI: Clinical Risk Assessment App
echo ===================================================
echo.

if exist "venv\Scripts\python.exe" (
    echo Using project virtual environment...
    ".\venv\Scripts\python.exe" -m streamlit run app2.py
) else (
    echo Virtual environment not found, attempting system Python...
    python -m streamlit run app2.py
)

pause
