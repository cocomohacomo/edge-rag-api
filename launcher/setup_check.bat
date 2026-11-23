@echo off
setlocal
cd /d "%~dp0.."
set "PY=%~dp0..\runtime\python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -c "import sqlite3,sys; print('Python:',sys.version); print('SQLite:',sqlite3.sqlite_version); c=sqlite3.connect(':memory:'); c.execute('create virtual table f using fts5(x)'); print('FTS5: OK')"
if errorlevel 1 (echo [NG] Python or SQLite FTS5 unavailable& exit /b 1)
"%PY%" -c "import pypdf; print('pypdf: OK')" 2>nul
if errorlevel 1 echo [INFO] pypdf not installed. TXT/MD search works; PDF import disabled.
echo [OK] Local prototype checks passed.
pause
