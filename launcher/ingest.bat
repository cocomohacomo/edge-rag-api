@echo off
setlocal
cd /d "%~dp0.."
set "PY=%~dp0..\runtime\python\python.exe"
if not exist "%PY%" set "PY=python"
if "%~1"=="" (echo Usage: ingest.bat ^<file-or-folder^>& exit /b 2)
"%PY%" rag\cli.py ingest %*
pause
