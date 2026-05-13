#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY=python3

# Python 3.10+ is required
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    echo "ERROR: Python 3.10 or higher is required to run LoRe."
    echo "Your default $PY is: $($PY --version 2>&1)"
    echo ""
    echo "To fix:"
    echo "Option A: Edit this run.sh file and change 'PY=python3' to"
    echo "point to your specific version (e.g. 'PY=python3.10')."
    echo "Option B: Install a newer version of Python and ensure it's in your PATH."
    exit 1
fi

if [ ! -d ".venv" ]; then
  $PY -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade .

python -m lore ui
