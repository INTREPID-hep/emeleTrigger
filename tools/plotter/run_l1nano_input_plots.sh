#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMELETRIGGER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

ROOT_FILE_DEFAULT="$WORKSPACE_DIR/HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8_L1NanoWithGenPropagated_20260212.root"
OUT_DIR_DEFAULT="$EMELETRIGGER_DIR/output/l1nano_inputs"
MAX_EVENTS_DEFAULT="-1"
TREE_DEFAULT="Events"

if [ ! -f "$ROOT_FILE_DEFAULT" ]; then
  shopt -s nullglob
  ROOT_CANDIDATES=("$WORKSPACE_DIR"/*.root)
  shopt -u nullglob
  if [ ${#ROOT_CANDIDATES[@]} -gt 0 ]; then
    ROOT_FILE_DEFAULT="${ROOT_CANDIDATES[0]}"
  fi
fi

ROOT_FILE="${1:-$ROOT_FILE_DEFAULT}"
OUT_DIR="${2:-$OUT_DIR_DEFAULT}"
MAX_EVENTS="${3:-$MAX_EVENTS_DEFAULT}"
TREE_NAME="${4:-$TREE_DEFAULT}"

if [ -n "${CONDA_ENV_NAME:-}" ]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
elif [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif [ -x "$WORKSPACE_DIR/.venv/bin/python" ]; then
  PYTHON_CMD=("$WORKSPACE_DIR/.venv/bin/python")
else
  PYTHON_CMD=(python)
fi

echo "Running L1Nano input variable plots"
echo "  ROOT file : $ROOT_FILE"
echo "  out dir   : $OUT_DIR"
echo "  max events: $MAX_EVENTS"
echo "  tree      : $TREE_NAME"

mkdir -p "$OUT_DIR"

MPLBACKEND=Agg "${PYTHON_CMD[@]}" "$SCRIPT_DIR/draw_variables.py" \
  --mode l1nano \
  --ifile "$ROOT_FILE" \
  --tree "$TREE_NAME" \
  --ofolder "$OUT_DIR" \
  --max-events "$MAX_EVENTS"

echo "Done. Generated files in: $OUT_DIR"
