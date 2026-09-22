#!/usr/bin/env bash
# Create the project virtualenv and install dependencies.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "Done. Activate with: source .venv/bin/activate"
