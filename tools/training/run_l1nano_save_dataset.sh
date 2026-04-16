#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMELETRIGGER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

ROOT_FILE_DEFAULT="$WORKSPACE_DIR/HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8_L1NanoWithGenPropagated_20260212.root"
OUT_FILE_DEFAULT="$EMELETRIGGER_DIR/datasets/l1nano_graphs_HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_20260212.pt"

if [ ! -f "$ROOT_FILE_DEFAULT" ]; then
  shopt -s nullglob
  ROOT_CANDIDATES=("$WORKSPACE_DIR"/*.root)
  shopt -u nullglob
  if [ ${#ROOT_CANDIDATES[@]} -gt 0 ]; then
    ROOT_FILE_DEFAULT="${ROOT_CANDIDATES[0]}"
  fi
fi

ROOT_FILE="${1:-$ROOT_FILE_DEFAULT}"
MAX_EVENTS="${2:--1}"
MAX_FILES="${3:-1}"
OUT_FILE="${4:-$OUT_FILE_DEFAULT}"

if [ -n "${CONDA_ENV_NAME:-}" ]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
elif [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(python)
fi

mkdir -p "$(dirname "$OUT_FILE")"

echo "Saving L1Nano dataset"
echo "  ROOT file : $ROOT_FILE"
echo "  max events: $MAX_EVENTS"
echo "  max files : $MAX_FILES"
echo "  output    : $OUT_FILE"

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/InputDataset.py" \
  --root_dir "$ROOT_FILE" \
  --tree_name Events \
  --max_files "$MAX_FILES" \
  --max_events "$MAX_EVENTS" \
  --save_path "$OUT_FILE" --debug

echo "Done. Dataset saved at: $OUT_FILE"
