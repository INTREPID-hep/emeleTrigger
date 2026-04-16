#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMELETRIGGER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_PATH_DEFAULT="$EMELETRIGGER_DIR/datasets/l1nano_graphs_HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_20260212.pt"

DATASET_PATH="${1:-$DATASET_PATH_DEFAULT}"
NUM_EXAMPLES="${2:-6}"
OUTPUT_PREFIX="${3:-}"

if [ -n "${CONDA_ENV_NAME:-}" ]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
elif [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("$PYTHON_BIN")
else
  PYTHON_CMD=(python)
fi

echo "Inspecting L1Nano dataset"
echo "  Dataset path  : $DATASET_PATH"
echo "  Num examples  : $NUM_EXAMPLES"
if [ -n "$OUTPUT_PREFIX" ]; then
  echo "  Output prefix : $OUTPUT_PREFIX"
fi

MPLBACKEND=Agg "${PYTHON_CMD[@]}" "$SCRIPT_DIR/inspect_dataset.py" \
  --dataset_path "$DATASET_PATH" \
  --num_examples "$NUM_EXAMPLES" \
  ${OUTPUT_PREFIX:+--output_prefix "$OUTPUT_PREFIX"} \
  --no_show

echo "Done!"
