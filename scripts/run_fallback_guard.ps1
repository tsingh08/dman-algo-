# run_fallback_guard.ps1 — wrapper invoked by the DManFallbackGuard scheduled
# task. Runs dman_algo.py --mode fallback-guard, which is deliberately
# independent of GitHub Actions entirely (see run_fallback_guard()'s
# docstring in dman_algo.py for the 2026-08-06 outage that motivated it,
# and the 2026-08-31 weekend/Monday-morning scheduler gap this task closes).
#
# GITHUB_TOKEN is pulled from the locally-authenticated `gh` CLI at runtime
# instead of being stored as a second copy of the secret anywhere.
#
# Installed via register_fallback_task.ps1 — do not run this standalone
# expecting persistent effects beyond a single check.

$ErrorActionPreference = "Continue"
Set-Location "C:\Users\singh"

$env:PYTHONIOENCODING = "utf-8"
$env:TELEGRAM_TOKEN    = [System.Environment]::GetEnvironmentVariable("TELEGRAM_TOKEN",   "User")
$env:TELEGRAM_CHAT_ID  = [System.Environment]::GetEnvironmentVariable("TELEGRAM_CHAT_ID", "User")

try {
    $env:GITHUB_TOKEN = (& gh auth token 2>$null)
} catch {
    $env:GITHUB_TOKEN = ""
}

$logFile   = "C:\Users\singh\dman_logs\fallback_guard_$(Get-Date -Format 'yyyy-MM-dd').log"
$ts        = Get-Date -Format "yyyy-MM-dd HH:mm"

if (-not (Test-Path "C:\Users\singh\dman_logs")) {
    New-Item -ItemType Directory -Path "C:\Users\singh\dman_logs" | Out-Null
}

"[$ts] ===== FALLBACK GUARD CHECK =====" | Out-File -FilePath $logFile -Append -Encoding utf8
try {
    $output = & py -X utf8 "C:\Users\singh\dman_algo.py" --mode fallback-guard 2>&1
    $output | Out-File -FilePath $logFile -Append -Encoding utf8
} catch {
    "[$ts] ERROR: $_" | Out-File -FilePath $logFile -Append -Encoding utf8
}
