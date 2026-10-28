# PowerShell launcher for Agentic K8s Scheduler Studio
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting Agentic K8s Scheduler Studio" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Opening http://localhost:8000 in your browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8000"
Write-Host ""
python backend\run.py --port 8000
