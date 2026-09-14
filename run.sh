#!/usr/bin/env bash
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
