@echo off
title Lithology Predictor Installer & Launcher
echo ========================================================
echo   LITHOLOGY PREDICTOR - SETUP AND RUN
echo ========================================================
echo.

echo 1. Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from python.org
    pause
    exit
)

echo 2. Installing/Updating Requirements (this may take a minute)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit
)

echo.
echo 3. Starting the App...
echo Opening your browser...
echo.
python -m streamlit run app.py

pause
