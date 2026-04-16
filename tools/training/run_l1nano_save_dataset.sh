#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

ROOT_FILE_DEFAULT="/lustre/ific.uv.es/ml/uovi156/data/HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8_L1NanoWithGenPropagated_20260212.root"
OUT_FILE_DEFAULT="/lustre/ific.uv.es/ml/uovi156/data/graphs/l1nano_graphs_HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_20260212.pt"
OUT_DIR_DEFAULT="/lustre/ific.uv.es/ml/uovi156/data/graphs/l1nano_graphs_prod"
PROD_DIR_DEFAULT="/lustre/ific.uv.es/ml/uovi156/data/prod"
TREE_DEFAULT="Events"
MAX_EVENTS_DEFAULT="-1"
MAX_FILES_PER_SAMPLE_DEFAULT="-1"
SAMPLES_DEFAULT=(B1 B2 B3 B4 S1 S2 S3 S4 S5)

if [ ! -f "$ROOT_FILE_DEFAULT" ]; then
  shopt -s nullglob
  ROOT_CANDIDATES=("$WORKSPACE_DIR"/*.root)
  shopt -u nullglob
  if [ ${#ROOT_CANDIDATES[@]} -gt 0 ]; then
    ROOT_FILE_DEFAULT="${ROOT_CANDIDATES[0]}"
  fi
fi

ROOT_FILE="$ROOT_FILE_DEFAULT"
MAX_EVENTS="$MAX_EVENTS_DEFAULT"
TREE_NAME="$TREE_DEFAULT"
MAX_FILES="1"
OUT_FILE="$OUT_FILE_DEFAULT"
OUT_DIR="$OUT_DIR_DEFAULT"
PROD_DIR="$PROD_DIR_DEFAULT"
MAX_FILES_PER_SAMPLE="$MAX_FILES_PER_SAMPLE_DEFAULT"
CONFIG_FILE=""
RUN_SCAN_MODE=false

show_help() {
  cat <<EOF
Usage:
  Single file mode (backward compatible):
    $0 <root_file> <max_events> <max_files> <out_file>

  Scan mode for prod folders B1..B4 and S1..S5:
    $0 --scan [--prod-dir DIR] [--out-dir DIR] [--max-events N] [--tree NAME] [--max-files-per-sample N]

Options:
  --scan                    Enable scan mode over B1..B4 and S1..S5 folders
  --prod-dir DIR            Base prod folder (default: $PROD_DIR_DEFAULT)
  --out-dir DIR             Output base folder for .pt files (default: $OUT_DIR_DEFAULT)
  --max-events N            Events per ROOT file passed to InputDataset.py (default: $MAX_EVENTS_DEFAULT)
  --tree NAME               TTree name (default: $TREE_DEFAULT)
  --max-files-per-sample N  Limit files per sample; -1 means all files (default: $MAX_FILES_PER_SAMPLE_DEFAULT)
  --config FILE             Optional YAML config for InputDataset.py (default: not passed)
  -h, --help                Show this help
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scan)
      RUN_SCAN_MODE=true
      shift
      ;;
    --prod-dir)
      PROD_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --max-events)
      MAX_EVENTS="$2"
      shift 2
      ;;
    --tree)
      TREE_NAME="$2"
      shift 2
      ;;
    --max-files-per-sample)
      MAX_FILES_PER_SAMPLE="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ "$RUN_SCAN_MODE" == false ]]; then
  ROOT_FILE="${POSITIONAL[0]:-$ROOT_FILE_DEFAULT}"
  MAX_EVENTS="${POSITIONAL[1]:-$MAX_EVENTS_DEFAULT}"
  MAX_FILES="${POSITIONAL[2]:-1}"
  OUT_FILE="${POSITIONAL[3]:-$OUT_FILE_DEFAULT}"
fi

CONFIG_FLAGS=()
if [[ -n "$CONFIG_FILE" ]]; then
  CONFIG_FLAGS=(--config "$CONFIG_FILE")
fi

if [ -n "${CONDA_ENV_NAME:-}" ]; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
elif [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD=("$PYTHON_BIN")
elif [ -x "$WORKSPACE_DIR/.venv/bin/python" ]; then
  PYTHON_CMD=("$WORKSPACE_DIR/.venv/bin/python")
else
  PYTHON_CMD=(python)
fi

echo "Saving L1Nano dataset"
if [[ "$RUN_SCAN_MODE" == false ]]; then
  mkdir -p "$(dirname "$OUT_FILE")"
  echo "  mode      : single file"
  echo "  ROOT file : $ROOT_FILE"
  echo "  max events: $MAX_EVENTS"
  echo "  max files : $MAX_FILES"
  echo "  tree      : $TREE_NAME"
  echo "  output    : $OUT_FILE"

  "${PYTHON_CMD[@]}" "$SCRIPT_DIR/InputDataset.py" \
    "${CONFIG_FLAGS[@]}" \
    --root_dir "$ROOT_FILE" \
    --tree_name "$TREE_NAME" \
    --max_files "$MAX_FILES" \
    --max_events "$MAX_EVENTS" \
    --save_path "$OUT_FILE" --debug

  echo "Done. Dataset saved at: $OUT_FILE"
  exit 0
fi

echo "  mode      : scan"
echo "  prod dir  : $PROD_DIR"
echo "  out dir   : $OUT_DIR"
echo "  max events: $MAX_EVENTS"
echo "  tree      : $TREE_NAME"
echo "  max files/sample: $MAX_FILES_PER_SAMPLE"

if [[ ! -d "$PROD_DIR" ]]; then
  echo "ERROR: prod directory not found: $PROD_DIR"
  exit 1
fi

mkdir -p "$OUT_DIR"

for sample in "${SAMPLES_DEFAULT[@]}"; do
  sample_dir="$PROD_DIR/$sample"
  if [[ ! -d "$sample_dir" ]]; then
    echo "[WARNING] sample folder not found, skipping: $sample_dir"
    continue
  fi

  shopt -s nullglob
  nano_files=("$sample_dir"/*_nano_*.root)
  shopt -u nullglob

  if [[ ${#nano_files[@]} -eq 0 ]]; then
    echo "[WARNING] no *_nano_*.root files in $sample_dir"
    continue
  fi

  echo "Processing sample $sample (${#nano_files[@]} nano files found)"
  processed=0
  failed=0

  for root_file in "${nano_files[@]}"; do
    if [[ "$MAX_FILES_PER_SAMPLE" -ge 0 && "$processed" -ge "$MAX_FILES_PER_SAMPLE" ]]; then
      break
    fi

    base_name="$(basename "$root_file" .root)"
    sample_out_dir="$OUT_DIR/$sample"
    out_file="$sample_out_dir/${base_name}.pt"
    mkdir -p "$sample_out_dir"

    echo "  -> $base_name"
    if "${PYTHON_CMD[@]}" "$SCRIPT_DIR/InputDataset.py" \
      "${CONFIG_FLAGS[@]}" \
      --root_dir "$root_file" \
      --tree_name "$TREE_NAME" \
      --max_files 1 \
      --max_events "$MAX_EVENTS" \
      --save_path "$out_file" --debug; then
      processed=$((processed + 1))
    else
      failed=$((failed + 1))
      echo "  [WARNING] failed for $base_name, continuing"
    fi
  done

  echo "  processed $processed files for $sample (failed: $failed)"
done

echo "Done. Scan outputs available in: $OUT_DIR"
