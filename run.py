#!/usr/bin/env python3
"""
LoRe bootstrap helper.

Generates novice-friendly launcher scripts:
- run.sh   (macOS/Linux)
- run.bat  (Windows)

Important: This script only writes local files and prints instructions.
The generated launcher scripts will create the venv and install dependencies.
"""

from __future__ import annotations

import argparse
import os
import platform
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WHEEL_REL_PATH = Path("wheels/ncbi_datasets_pylib-1.0.0-py3-none-any.whl")

RUN_SH = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY=python3
if [ ! -d ".venv" ]; then
  $PY -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade ./wheels/ncbi_datasets_pylib-1.0.0-py3-none-any.whl
python -m pip install --upgrade .

python -m lore ui
"""

RUN_BAT = """@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\\Scripts\\python.exe" (
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

call .venv\\Scripts\\activate.bat
python -m pip install --upgrade pip
python -m pip install --upgrade .\\wheels\\ncbi_datasets_pylib-1.0.0-py3-none-any.whl
python -m pip install --upgrade .

python -m lore ui
"""


def write_file(path: Path, content: str, force: bool, dry_run: bool) -> str:
    """Write file unless it exists and force=False. Returns action label."""
    if path.exists() and not force:
        return "kept"

    if dry_run:
        return "would-write"

    path.write_text(content, encoding="utf-8", newline="\n")
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

    sh_state = write_file(run_sh_path, RUN_SH, force=args.force, dry_run=args.dry_run)
    bat_state = write_file(run_bat_path, RUN_BAT, force=args.force, dry_run=args.dry_run)
    make_executable(run_sh_path, dry_run=args.dry_run)

    print("LoRē bootstrap")
    print("==============")
    print(f"Repository: {REPO_ROOT}")
    print(f"run.sh: {sh_state}")
    print(f"run.bat: {bat_state}")

    wheel_path = REPO_ROOT / WHEEL_REL_PATH
    if wheel_path.exists():
        print(f"Wheel found: {WHEEL_REL_PATH}")
    else:
        print(f"WARNING: Wheel not found at {WHEEL_REL_PATH}")
        print("         Update the wheel path in run.py or add the wheel file.")

    current_os = platform.system().lower()
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
