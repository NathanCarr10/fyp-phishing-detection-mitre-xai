@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_reproducible_pipeline.ps1" %*
exit /b %ERRORLEVEL%
