# register_fallback_task.ps1 — one-time setup, run manually (not from CI):
#     powershell -ExecutionPolicy Bypass -File scripts\register_fallback_task.ps1
#
# Registers a Windows Task Scheduler task that runs run_fallback_guard.ps1
# every 15 minutes, weekdays, wake-from-sleep included. This is the local,
# GitHub-Actions-INDEPENDENT backstop referenced in run_fallback_guard()'s
# docstring in dman_algo.py — confirmed necessary live 2026-08-06 (a
# multi-hour GitHub platform outage) and again 2026-08-31 (every scheduled
# GitHub Actions trigger in the repo went silent from Saturday night through
# Monday midday with no incident reported). Deliberately does not touch
# GitHub's runner system at all so a GitHub-side failure can't also take
# down the thing meant to catch it.
#
# Re-run this script any time to update the registered task in place
# (it replaces the existing one with -Force).

$taskName    = "DManFallbackGuard"
$taskPath    = "\DmanAlgo\"
$scriptPath  = Join-Path $PSScriptRoot "run_fallback_guard.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$scriptPath`""

$trigger = New-ScheduledTaskTrigger -Once -At "07:00" `
    -RepetitionInterval (New-TimeSpan -Minutes 15) `
    -RepetitionDuration (New-TimeSpan -Hours 9)
$trigger.DaysOfWeek = 62   # Mon(2)+Tue(4)+Wed(8)+Thu(16)+Fri(32) bitmask — weekdays only

$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $taskName -TaskPath $taskPath `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Local, GitHub-Actions-independent health check. Detects a dark GitHub scheduler/outage and runs today's scan locally so there's no silent gap in coverage. See run_fallback_guard() in dman_algo.py." `
    -Force | Out-Null

Write-Output "Registered '$taskPath$taskName' — every 15 min, weekdays 7:00am-4:00pm local, wake-to-run enabled."
