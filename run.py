#!/usr/bin/env python3
"""
LoRē bootstrap helper.

Generates novice-friendly launcher scripts:
- run.sh   (macOS/Linux)
- run.bat  (Windows)

Important: This script only writes local files and prints instructions.
The generated launcher scripts will create the venv and install dependencies.
"""

import argparse
import os
import platform
import stat
import sys
import time
from pathlib import Path

# --- Formatting Helpers ---
def print_header(title: str):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50 + "\n")

if sys.version_info < (3, 11):
    print_header("LoRē bootstrap")
    print("WARNING: Python 3.11 or higher is required to run this script.")
    print("Your current Python version is: %s", {sys.version.split()[0]})
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent


RUN_SH = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 1. Check Python version (requires 3.11+)
PY=""
for candidate in python3 python3.13 python3.12 python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    print_header("ERROR: Python 3.11 or higher is required to run LoRe.")
    echo "No valid Python 3.11+ executable was found in your PATH."
    echo ""
    echo "To fix:"
    echo "Option A: Install a newer version of Python and ensure it's in your PATH."
    echo "Option B: If you think this is a mistake, edit this run.sh file and change"
    echo "'PY=your_python_here' to point to your specific binary."
    exit 1
fi

# 2. Set up virtual environment
if [ ! -d ".venv" ]; then
  $PY -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade .

python -m lore ui
"""


RUN_BAT = """@echo off
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
if not exist ".venv\\Scripts\\python.exe" (
    %PY_CMD% -m venv .venv
)

call .venv\\Scripts\\activate.bat

:: 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
if %errorlevel% neq 0 ( echo ERROR: pip upgrade failed ^& pause ^& exit /b 1 )

python -m pip install --upgrade .
if %errorlevel% neq 0 ( echo ERROR: package install failed ^& pause ^& exit /b 1 )

:: 4. Launch the UI
python -m lore ui
pause
"""


def write_file(path: Path, content: str, force: bool, dry_run: bool, newline: str = "\n") -> str:
    """Write file unless it exists and force=False. Returns action label."""
    if path.exists() and not force:
        return "kept"

    if dry_run:
        return "would-write"

    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(content)
    return "written"


def make_executable(path: Path, dry_run: bool) -> None:
    """Set execute bit on Unix-like systems."""
    if dry_run or os.name == "nt" or not path.exists():
        return

    current = path.stat().st_mode
    path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate local launcher scripts for LoRē (run.sh / run.bat)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing run.sh and run.bat.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without making changes.",
    )
    args = parser.parse_args()

    run_sh_path = REPO_ROOT / "run.sh"
    run_bat_path = REPO_ROOT / "run.bat"

    sh_state = write_file(
        run_sh_path, RUN_SH, force=args.force, dry_run=args.dry_run,
    )
    bat_state = write_file(
        run_bat_path, RUN_BAT, force=args.force, dry_run=args.dry_run, newline="\r\n",
    )
    make_executable(run_sh_path, dry_run=args.dry_run)

    print_header("LoRē bootstrap")

    print(f"Repository : {REPO_ROOT}")
    print(f"run.sh     : {sh_state}")
    print(f"run.bat    : {bat_state}")

    current_os = platform.system().lower()
    print(f"\nCurrent OS   : {current_os}")

    print_header("Setup complete!")
    print("\nNow, you can run LoRē with the following file:")
    if "windows" in current_os:
        print("  Run: .\\run.bat")
    else:
        print("  Run: ./run.sh")

    print("\nThis will verify the virtual environment and dependencies, then launch the UI.")
    print("\nYou may want to create a shortcut to it for easy access in the future.")

    print("\n\nNote:")
    print("  If you do not want automatic virtual environment setup or pip upgrades,"
    " please refer to the 'Manual setup' section of the README.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
