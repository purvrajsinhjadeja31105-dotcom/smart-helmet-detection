@echo off
REM ================================================================
REM  Smart Helmet Detection - demo launcher
REM  Use: drag an image or video file onto this file,
REM       or double-click it and paste / drag the file path.
REM  Works offline. Results: outputs\<run_id>\ and reports\<run_id>.json
REM ================================================================
title Smart Helmet Detection - Demo
cd /d "%~dp0backend"

if "%~1"=="" (
    echo.
    echo  Drag an image or video into this window and press Enter:
    set /p FILE=  File:
) else (
    set "FILE=%~1"
)
set "FILE=%FILE:"=%"

if not exist "%FILE%" (
    echo.
    echo  File not found: %FILE%
    pause
    exit /b 1
)

echo.
echo  Processing "%FILE%" ... (first run loads the models, about 30 seconds)
echo.
"%~dp0.venv\Scripts\python.exe" -m app.cli "%FILE%" --open
echo.
echo  Done. The result window has opened. Full details are in the outputs folder shown above.
pause
