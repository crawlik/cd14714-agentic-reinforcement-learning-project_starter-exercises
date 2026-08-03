#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage 1 on the Mac. Everything runs from this directory; nothing is written
# outside it.
#
#   bash run_stage1_mac.sh check     preflight only
#   bash run_stage1_mac.sh smoke     ~2 min pipeline test, throwaway adapter
#   bash run_stage1_mac.sh train     the real run (2-3 h on MPS)
#   bash run_stage1_mac.sh short     3 epochs, to sanity-check hyperparameters
# ---------------------------------------------------------------------------
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${MEETMIND_VENV:-$PROJECT_DIR/../../agent-course-env}"
cd "$PROJECT_DIR"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "No venv at $VENV"
  echo "Set one explicitly:  MEETMIND_VENV=/path/to/venv bash run_stage1_mac.sh $*"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# MPS still has op-coverage gaps; the fallback keeps one missing kernel from
# aborting a multi-hour run instead of quietly running that op on CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

MODE="${1:-check}"

# Everything is teed into logs/ so the run can be reviewed afterwards without
# anyone having to copy terminal output around by hand.
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/${MODE}_${STAMP}.log"

{
  echo "project : $PROJECT_DIR"
  echo "python  : $(which python)"
  echo "mode    : $MODE"
  echo "started : $(date)"
  echo
} | tee "$LOG"

run() { "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

case "$MODE" in
  check)
    run python env_check.py
    ;;
  smoke)
    run python starter_sft.py --smoke
    ;;
  short)
    run caffeinate -i python starter_sft.py --epochs 3 --output models/sft_short_run
    ;;
  train)
    # caffeinate -i: a closed lid should not kill two hours of training
    run caffeinate -i python starter_sft.py
    ;;
  all)
    # preflight, then the smoke test only if preflight passed
    run python env_check.py && run python starter_sft.py --smoke
    ;;
  *)
    echo "unknown mode '$MODE' (check | smoke | short | train | all)" | tee -a "$LOG"
    exit 1
    ;;
esac

STATUS=$?
{
  echo
  echo "finished: $(date)  exit=$STATUS"
} | tee -a "$LOG"
echo
echo "log written to $PROJECT_DIR/$LOG"
exit "$STATUS"
