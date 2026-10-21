#!/usr/bin/env bash
# Auris - Linux / macOS setup wrapper
# Usage: bash setup.sh

set -e
cd "$(dirname "$0")"

# Pick the first interpreter that is new enough (3.10+). The order prefers
# the versions that are tested on ARM64.
PYTHON=""
for candidate in python3.12 python3.11 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.10+ not found. Please install Python 3.10 or newer first."
    exit 1
fi

echo "Using Python: $PYTHON ($("$PYTHON" --version 2>&1))"

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating virtual environment in .venv..."
    "$PYTHON" -m venv .venv
fi

".venv/bin/python" setup.py
