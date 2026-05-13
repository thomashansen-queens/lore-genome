@echo off
setlocal

cd /d "%~dp0"

:: 1. Detect Python executable and check version
set "PY_CMD=python"
where py >nul 2>nul && set "PY_CMD=py -3"

%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.10 or higher is required to run LoRe.
    echo Your current version is:
    %PY_CMD% --version
    echo.
    echo To fix:
    echo Option A: Edit this run.bat file and change 'set "PY_CMD=python"' to 
    echo point to your specific version, e.g. 'set "PY_CMD=py -3.10"'.
    echo Option B: Install a newer version of Python and ensure it's in your PATH.
    pause
    exit /b 1
)

:: 2. Create and activate virtual environment
if not exist ".venv\Scripts\python.exe" (
    %PY_CMD% -m venv .venv
)

call .venv\Scripts\activate.bat

:: 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
if %errorlevel% neq 0 ( echo ERROR: pip upgrade failed ^& pause ^& exit /b 1 )

python -m pip install --upgrade .
if %errorlevel% neq 0 ( echo ERROR: package install failed ^& pause ^& exit /b 1 )

:: 4. Launch the UI
python -m lore ui
pause
