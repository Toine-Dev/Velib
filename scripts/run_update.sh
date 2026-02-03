#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-"../.venv/bin/python"}
exec "$PY" scripts/update_data.py "$@"
