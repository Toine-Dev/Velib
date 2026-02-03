#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-"../.venv/bin/python"}
DATA=${1:-"comptage_velo_donnees_compteurs.csv"}
MODEL=${2:-"artifacts/model.pkl"}
OUT=${3:-"artifacts/eval_metrics.json"}
exec "$PY" models/evaluation.py --data "$DATA" --model "$MODEL" --out "$OUT" "$@"
