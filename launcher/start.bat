@echo off
setlocal
cd /d "%~dp0.."
set "PY=%~dp0..\runtime\python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" rag\cli.py chat
