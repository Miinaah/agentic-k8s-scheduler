@echo off
echo ========================================================
echo   Starting Agentic K8s Scheduler Studio
echo ========================================================
echo.
echo Opening browser at http://localhost:8000 ...
start http://localhost:8000
echo.
python backend\run.py --port 8000
pause
