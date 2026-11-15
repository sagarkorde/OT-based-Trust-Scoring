# OT-Shield setup for Windows 11 + NVIDIA laptop GPU (laptop profile).
# Run from the folder that holds this script (e.g. C:\Users\<you>\ot-shield):
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Safe to re-run: every step checks before acting. Writes logs\setup_<time>.log and logs\setup_report.json.
#
# What it does
#   1. Records the machine (Windows, CPU, RAM, GPU, driver, free disk).
#   2. Checks git, conda (Anaconda/Miniconda) and Ollama; installs a missing one with winget (asks first).
#   3. Clones or updates the project repository into .\repo.
#   4. Creates conda env 'otshield' (Python 3.11): CUDA PyTorch, project requirements, beir (--no-deps).
#   5. Clones the four benchmarks at their pinned commits (python tasks.py benchmarks).
#   6. Pulls the Ollama models and creates fixed-context tags (ots-*), with flash attention + q8 KV cache.
#   7. Checks Hugging Face login and access to the gated models.
#   8. Runs the unit tests and the CPU smoke pipeline.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Join-Path $Root "repo"
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Start-Transcript -Path (Join-Path $Logs "setup_$Stamp.log") | Out-Null
$Report = [ordered]@{ started = (Get-Date).ToString("s"); steps = [ordered]@{} }
function Step($name, $ok, $detail) {
    $Report.steps[$name] = [ordered]@{ ok = $ok; detail = "$detail" }
    $mark = if ($ok) { "OK  " } else { "FAIL" }
    Write-Host "[$mark] $name  $detail"
}
function Save-Report { $Report.finished = (Get-Date).ToString("s"); $Report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $Logs "setup_report.json") }
function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

$RepoUrl = "https://github.com/sagarkorde/OT-based-Trust-Scoring"
$EnvName = "otshield"

try {
    # ---------------------------------------------------------------- 1. machine
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    $ramGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $freeC = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
    $gpu = "nvidia-smi not found"
    if (Have "nvidia-smi") { $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) -join "; " }
    $Report.machine = [ordered]@{ os = $os.Caption; build = $os.BuildNumber; cpu = $cpu; ram_gb = $ramGB; free_c_gb = $freeC; gpu = $gpu }
    Step "machine" ($gpu -notlike "*not found*") "$cpu | $ramGB GB RAM | $gpu | C: free $freeC GB"
    if ($freeC -lt 100) { Write-Warning "Less than 100 GB free on C:. Models + datasets need ~100 GB over the whole study." }

    # ---------------------------------------------------------------- 2. tools
    function Ensure-Tool($cmd, $wingetId) {
        if (Have $cmd) { return $true }
        $ans = Read-Host "$cmd is not installed. Install it now with winget ($wingetId)? [y/N]"
        if ($ans -eq "y") {
            winget install --id $wingetId -e --accept-source-agreements --accept-package-agreements
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
        }
        return (Have $cmd)
    }
    $gitOk = Ensure-Tool "git" "Git.Git"
    Step "git" $gitOk $(if ($gitOk) { (git --version) } else { "missing" })

    $conda = $null
    foreach ($c in @((Get-Command conda.exe -ErrorAction SilentlyContinue).Source,
                     "$env:USERPROFILE\anaconda3\Scripts\conda.exe", "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
                     "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe", "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe",
                     "C:\ProgramData\anaconda3\Scripts\conda.exe", "C:\ProgramData\miniconda3\Scripts\conda.exe")) {
        if ($c -and (Test-Path $c)) { $conda = $c; break }
    }
    Step "conda" ([bool]$conda) $(if ($conda) { $conda } else { "not found: install Miniconda (winget install Anaconda.Miniconda3) and re-run" })
    if (-not $conda) { throw "conda missing" }

    $ollamaOk = Ensure-Tool "ollama" "Ollama.Ollama"
    Step "ollama" $ollamaOk $(if ($ollamaOk) { (ollama --version) -join " " } else { "missing" })

    # ---------------------------------------------------------------- 3. repository
    if (Test-Path (Join-Path $Repo ".git")) { git -C $Repo pull --ff-only } else { git clone $RepoUrl $Repo }
    Step "repo" (Test-Path (Join-Path $Repo "tasks.py")) (git -C $Repo log -1 --format="%h %s")

    # ---------------------------------------------------------------- 4. python environment
    $envList = & $conda env list
    if (-not ($envList -match "^\s*$EnvName\s")) { & $conda create -y -n $EnvName python=3.11 }
    # locate the env from conda's JSON listing (`conda run` may print nothing on Windows)
    $envDir = ((& $conda env list --json) | Out-String | ConvertFrom-Json).envs |
              Where-Object { (Split-Path $_ -Leaf) -eq $EnvName } | Select-Object -First 1
    $Py = if ($envDir) { Join-Path $envDir "python.exe" } else { $null }
    if (-not $Py -or -not (Test-Path $Py)) { throw "python.exe for env '$EnvName' not found (envs: $((& $conda env list) -join '; '))" }
    Write-Host "Python: $Py"
    & $Py -m pip install --upgrade pip
    & $Py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    & $Py -m pip install -r (Join-Path $Repo "requirements-windows.txt")
    & $Py -m pip install --no-deps "beir==2.0.0"
    $cuda = ((& $Py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')") | Select-Object -Last 1).Trim()
    Step "python-env" ($cuda -like "*True*") "$Py | torch $cuda"
    $Report.python = $Py

    # ---------------------------------------------------------------- 5. benchmarks
    Push-Location $Repo
    & $Py tasks.py benchmarks
    Pop-Location
    Step "benchmarks" (Test-Path (Join-Path $Repo "ext\agentdojo\.git")) "pinned commits checked out"

    # ---------------------------------------------------------------- 6. Ollama models
    # flash attention + 8-bit KV cache keep a 16k context of an 8B Q4 model inside 8 GB of VRAM
    [Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
    $env:OLLAMA_FLASH_ATTENTION = "1"; $env:OLLAMA_KV_CACHE_TYPE = "q8_0"; $env:OLLAMA_MAX_LOADED_MODELS = "1"; $env:OLLAMA_NUM_PARALLEL = "1"
    Write-Host "Restarting Ollama so it picks up the settings above..."
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
    $app = "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
    if (Test-Path $app) { Start-Process $app } else { Start-Process ollama -ArgumentList "serve" -WindowStyle Minimized }
    Start-Sleep -Seconds 6

    $tags = [ordered]@{
        "ots-llama3.1-8b-16k"  = @("llama3.1:8b", 16384)                       # E3 agents, G2, G3
        "ots-llama3.1-8b-4k"   = @("llama3.1:8b", 4096)                        # E1 known-answer, E2, E4 paraphrase
        "ots-qwen2.5-7b-4k"    = @("qwen2.5:7b", 4096)                         # E2 second LLM
        "ots-llama2-7b-4k"     = @("llama2:7b-chat", 4096)                     # G1 (PoisonedRAG reports LLaMA-2-7B)
        "ots-mistral-7b-v02-4k" = @("mistral:7b-instruct-v0.2-q4_K_M", 4096)   # G4 (HarmBench reports Mistral 7B)
    }
    $mf = Join-Path $Logs "modelfiles"; New-Item -ItemType Directory -Force -Path $mf | Out-Null
    $made = @()
    foreach ($t in $tags.Keys) {
        $base = $tags[$t][0]; $ctx = $tags[$t][1]
        ollama pull $base
        $file = Join-Path $mf "$t.Modelfile"
        "FROM $base`nPARAMETER num_ctx $ctx`nPARAMETER temperature 0" | Set-Content -Encoding ASCII $file
        ollama create $t -f $file
        $made += $t
    }
    $list = (ollama list) -join "`n"
    $allTags = ($made | Where-Object { $list -notmatch [regex]::Escape($_) }).Count -eq 0
    Step "ollama-models" $allTags ($made -join ", ")

    # ---------------------------------------------------------------- 7. Hugging Face access
    $hf = & $Py -c @"
from huggingface_hub import model_info, whoami
try:
    print('user=' + whoami()['name'])
except Exception as e:
    print('user=NOT-LOGGED-IN')
for m in ['meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-Guard-3-8B', 'cais/HarmBench-Llama-2-13b-cls', 'BAAI/bge-base-en-v1.5']:
    try:
        model_info(m); print(m + '=ok')
    except Exception as e:
        print(m + '=NO-ACCESS (' + type(e).__name__ + ')')
"@
    $hfText = ($hf -join " | ")
    Step "huggingface" ($hfText -notmatch "NOT-LOGGED-IN|NO-ACCESS") $hfText
    if ($hfText -match "NOT-LOGGED-IN|NO-ACCESS") {
        Write-Warning "Gated models: accept the licences on huggingface.co for meta-llama/Llama-3.1-8B-Instruct and meta-llama/Llama-Guard-3-8B, then log in with:  & `"$Py`" -c `"from huggingface_hub import login; login()`"  (paste a read token), and re-run this script."
    }

    # ---------------------------------------------------------------- 8. tests + smoke
    Push-Location $Repo
    $env:PYTHONUTF8 = "1"
    & $Py tasks.py test
    $testOk = ($LASTEXITCODE -eq 0)
    Step "unit-tests" $testOk "python tasks.py test"
    & $Py tasks.py smoke --profile laptop
    $smokeOk = ($LASTEXITCODE -eq 0)
    Step "smoke" $smokeOk "python tasks.py smoke --profile laptop"
    Pop-Location
}
catch {
    Step "aborted" $false $_.Exception.Message
}
finally {
    Save-Report
    Stop-Transcript | Out-Null
    Write-Host "`nReport: $(Join-Path $Logs 'setup_report.json')"
}
