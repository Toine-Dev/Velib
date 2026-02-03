#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-"../.venv/bin/python"}
DATA=${1:-"comptage_velo_donnees_compteurs.csv"}
MODEL_OUT=${2:-"artifacts/model.pkl"}
exec "$PY" models/train.py --data "$DATA" --model-out "$MODEL_OUT" "$@"
