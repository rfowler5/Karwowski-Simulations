# Run benchmark_full_grid.py 5 times consecutively (for README data).
# Run this in PowerShell from the project root: .\run_benchmark_5_times.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

foreach ($i in 1..5) {
    Write-Host "`n========== Run $i of 5 ==========" -ForegroundColor Cyan
    python benchmarks/benchmark_full_grid.py
    if ($LASTEXITCODE -ne 0) { throw "Run $i exited with $LASTEXITCODE" }
}
Write-Host "`n========== All 5 runs complete ==========" -ForegroundColor Green
