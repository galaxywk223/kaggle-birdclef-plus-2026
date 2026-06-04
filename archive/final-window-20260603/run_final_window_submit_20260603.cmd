@echo off
setlocal
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts\final_window_submit_20260603.ps1" >> "logs\final_window_submit_20260603_task_stdout.log" 2>> "logs\final_window_submit_20260603_task_stderr.log"
exit /b %ERRORLEVEL%
