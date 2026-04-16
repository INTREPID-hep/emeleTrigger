#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMELETRIGGER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

ROOT_FILE_DEFAULT="$WORKSPACE_DIR/HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8_L1NanoWithGenPropagated_20260212.root"
OUT_PREFIX_DEFAULT="$EMELETRIGGER_DIR/datasets/l1nano_evt0"

if [ ! -f "$ROOT_FILE_DEFAULT" ]; then
  shopt -s nullglob
  ROOT_CANDIDATES=("$WORKSPACE_DIR"/*.root)
  shopt -u nullglob
  if [ ${#ROOT_CANDIDATES[@]} -gt 0 ]; then
    ROOT_FILE_DEFAULT="${ROOT_CANDIDATES[0]}"
  fi
fi

ROOT_FILE="${1:-$ROOT_FILE_DEFAULT}"
PLOT_IDX="${2:-0}"
MAX_EVENTS="${3:-5}"
OUT_PREFIX="${4:-$OUT_PREFIX_DEFAULT}"

if [ -n "${CONDA_ENV_NAME:-}" ]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
elif [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(python)
fi

echo "Running L1Nano visualization"
echo "  ROOT file   : $ROOT_FILE"
echo "  plot idx    : $PLOT_IDX"
echo "  max events  : $MAX_EVENTS"
echo "  out prefix  : $OUT_PREFIX"

MPLBACKEND=Agg "${PYTHON_CMD[@]}" "$SCRIPT_DIR/InputDataset.py" \
  --root_dir "$ROOT_FILE" \
  --tree_name Events \
  --max_files 1 \
  --max_events "$MAX_EVENTS" \
  --inspect_event \
  --max_print_stubs 10 \
  --max_print_edges 12 \
  --plot_stub_edge_info \
  --plot_example \
  --show_edge_attr_labels \
  --plot_idx "$PLOT_IDX" \
  --plot_output_prefix "$OUT_PREFIX"

echo "Done. Output files:"
echo "  ${OUT_PREFIX}_stub_edge.png"
echo "  ${OUT_PREFIX}_graph.png"
