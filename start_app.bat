@echo off
echo ============================================================
echo Food Shelf Life Predictor
echo ============================================================
echo.
echo Starting server...
echo.

cd /d "%~dp0"
python app_flask.py

pause
