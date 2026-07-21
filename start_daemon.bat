@echo off
rem DMan always-on daemon launcher — auto-restarts on crash.
rem Auto-start: Task Scheduler > Create Basic Task > "At log on" > start this file.
cd /d %~dp0
:loop
py -3 dman_daemon.py
echo.
echo Daemon exited — restarting in 30s (press Ctrl+C to stop)
timeout /t 30
goto loop
