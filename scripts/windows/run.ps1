# Run one OT-Shield task with the laptop profile, in the conda env made by setup.ps1.
#     powershell -ExecutionPolicy Bypass -File .\run.ps1 gates
#     powershell -ExecutionPolicy Bypass -File .\run.ps1 gates G1 G4      (only some gates)
#     powershell -ExecutionPolicy Bypass -File .\run.ps1 e1
# Tasks: test, smoke, gates, gate-check, e1, e2, e3, e4, latency, tables, figures, refs, all
# Output streams to the console and to repo\logs\<task>_<time>.log; results land in repo\results\.
# The laptop profile caps the run at 50 GB RAM (OT_MAX_RAM_GB) and loads models in 4-bit.
param([Parameter(Mandatory = $true)][string]$Task, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Join-Path $Root "repo"
$Py = (Get-Content (Join-Path $Root "logs\setup_report.json") -Raw | ConvertFrom-Json).python
if (-not $Py) { throw "Run setup.ps1 first." }
$env:PYTHONUTF8 = "1"
Push-Location $Repo
try {
    git pull --ff-only | Out-Null           # pick up fixes pushed to GitHub
    $extra = @()
    if ($Task -eq "gates" -and $Rest) { $extra = @("--", "--only") + $Rest }
    & $Py tasks.py $Task --profile laptop @extra
    exit $LASTEXITCODE
}
finally { Pop-Location }
