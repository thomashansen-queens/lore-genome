@echo off
setlocal

cd /d "%~dp0"

:: 1. Detect Python executable and check version
set "PY_CMD=python"
where py >nul 2>nul && set "PY_CMD=py -3"

%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 or higher is required to run LoRe.
    echo Your current version is:
    %PY_CMD% --version
    echo.
    echo To fix:
    echo Option A: Install a newer version of Python and ensure it's in your PATH.
    echo Option B: If you think this is a mistake, edit this run.bat file and change
    echo 'set "PY_CMD=python"' to point to your specific version,
    echo e.g. 'set "PY_CMD=py -3.11"'.
    pause
    exit /b 1
)

:: 2. Create and activate virtual environment
if not exist ".venv\Scripts\python.exe" (
    %PY_CMD% -m venv .venv
)

call .venv\Scripts\activate.bat

if not exist delete_this_when_updating.txt (
	:: 3. Upgrade pip and install dependencies
	python -m pip install --upgrade pip
	if %errorlevel% neq 0 ( echo ERROR: pip upgrade failed ^& pause ^& exit /b 1 )

	python -m pip install --upgrade .
	if %errorlevel% neq 0 ( echo ERROR: package install failed ^& pause ^& exit /b 1 )

	echo. > delete_this_when_updating.txt
)

:: 4. Launch the UI
python -m lore ui --port 8080
pause
