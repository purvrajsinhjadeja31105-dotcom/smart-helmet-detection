@echo off
REM ================================================================
REM  Smart Helmet Detection - start the web app
REM  Double-click this file. The browser opens http://localhost:8000
REM  when the models are loaded. Close this window to stop the app.
REM ================================================================
title Smart Helmet Detection - Web App
cd /d "%~dp0backend"
echo.
echo  Starting Smart Helmet Detection... (loading the AI models takes about 30 seconds)
echo  The browser will open automatically. Keep this window open while you use the app.
echo.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/v1/health -TimeoutSec 2 | Out-Null; Start-Process 'http://localhost:8000'; break}catch{Start-Sleep -Seconds 2}}"
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
pause
