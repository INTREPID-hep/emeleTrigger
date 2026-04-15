#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMELETRIGGER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

ROOT_FILE_DEFAULT="$WORKSPACE_DIR/HTo2LongLivedTo4mu_MH-125_MFF-12_CTau-900mm_TuneCP5_14TeV-pythia8_L1NanoWithGenPropagated_20260212.root"
OUT_DIR_DEFAULT="$EMELETRIGGER_DIR/output/l1nano_inputs"
PROD_DIR_DEFAULT="/lustre/ific.uv.es/ml/uovi156/data/prod"
MAX_EVENTS_DEFAULT="-1"
TREE_DEFAULT="Events"
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

ROOT_FILE="${1:-$ROOT_FILE_DEFAULT}"
OUT_DIR="${2:-$OUT_DIR_DEFAULT}"
MAX_EVENTS="${3:-$MAX_EVENTS_DEFAULT}"
TREE_NAME="${4:-$TREE_DEFAULT}"
RUN_SCAN_MODE=false
PROD_DIR="$PROD_DIR_DEFAULT"
MAX_FILES_PER_SAMPLE="$MAX_FILES_PER_SAMPLE_DEFAULT"

show_help() {
  cat <<EOF
Usage:
  Single file mode (backward compatible):
    $0 <root_file> <out_dir> <max_events> <tree_name>

  Scan mode for prod folders B1..B4 and S1..S5:
    $0 --scan [--prod-dir DIR] [--out-dir DIR] [--max-events N] [--tree NAME] [--max-files-per-sample N]

Options:
  --scan                    Enable scan mode over B1..B4 and S1..S5 folders
  --prod-dir DIR            Base prod folder (default: $PROD_DIR_DEFAULT)
  --out-dir DIR             Output base folder (default: $OUT_DIR_DEFAULT)
  --max-events N            Events per ROOT file passed to draw_variables.py (default: $MAX_EVENTS_DEFAULT)
  --tree NAME               TTree name (default: $TREE_DEFAULT)
  --max-files-per-sample N  Limit files per sample; -1 means all files (default: $MAX_FILES_PER_SAMPLE_DEFAULT)
  -h, --help                Show this help
EOF
}

if [[ "${1:-}" == "--scan" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  RUN_SCAN_MODE=true
fi

if [[ "$RUN_SCAN_MODE" == true ]]; then
  ROOT_FILE=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --scan)
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
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
    esac
  done
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

echo "Running L1Nano input variable plots"
if [[ "$RUN_SCAN_MODE" == false ]]; then
  echo "  mode      : single file"
  echo "  ROOT file : $ROOT_FILE"
  echo "  out dir   : $OUT_DIR"
  echo "  max events: $MAX_EVENTS"
  echo "  tree      : $TREE_NAME"
else
  echo "  mode      : scan"
  echo "  prod dir  : $PROD_DIR"
  echo "  out dir   : $OUT_DIR"
  echo "  max events: $MAX_EVENTS"
  echo "  tree      : $TREE_NAME"
  echo "  max files/sample: $MAX_FILES_PER_SAMPLE"
fi

mkdir -p "$OUT_DIR"

if [[ "$RUN_SCAN_MODE" == false ]]; then
  MPLBACKEND=Agg "${PYTHON_CMD[@]}" "$SCRIPT_DIR/draw_variables.py" \
    --mode l1nano \
    --ifile "$ROOT_FILE" \
    --tree "$TREE_NAME" \
    --ofolder "$OUT_DIR" \
    --max-events "$MAX_EVENTS"

  echo "Done. Generated files in: $OUT_DIR"
  exit 0
fi

if [[ ! -d "$PROD_DIR" ]]; then
  echo "ERROR: prod directory not found: $PROD_DIR"
  exit 1
fi

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
  for root_file in "${nano_files[@]}"; do
    if [[ "$MAX_FILES_PER_SAMPLE" -ge 0 && "$processed" -ge "$MAX_FILES_PER_SAMPLE" ]]; then
      break
    fi

    base_name="$(basename "$root_file" .root)"
    file_out_dir="$OUT_DIR/$sample/$base_name"
    mkdir -p "$file_out_dir"

    echo "  -> $base_name"
    MPLBACKEND=Agg "${PYTHON_CMD[@]}" "$SCRIPT_DIR/draw_variables.py" \
      --mode l1nano \
      --ifile "$root_file" \
      --tree "$TREE_NAME" \
      --ofolder "$file_out_dir" \
      --max-events "$MAX_EVENTS"

    processed=$((processed + 1))
  done

  echo "  processed $processed files for $sample"
done

echo "Done. Scan outputs available in: $OUT_DIR"
