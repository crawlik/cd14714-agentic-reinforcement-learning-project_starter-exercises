#!/usr/bin/env bash
# Read-only status report for the Stage 3 DPO evaluation run.
# Usage: bash .claude/skills/eval-status/eval_status.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
LOGDIR="project/logs"
[ -d "$LOGDIR" ] || LOGDIR="logs"

LOG=$(ls -t "$LOGDIR"/dpo_eval_*.log 2>/dev/null | head -1)
if [ -z "${LOG:-}" ]; then
  echo "no dpo_eval_*.log found in $LOGDIR"
  exit 0
fi

if pgrep -f starter_agentic_rlft >/dev/null 2>&1; then ALIVE=RUNNING; else ALIVE=NOT-RUNNING; fi

L=$(tr '\r' '\n' < "$LOG" | grep -c "complete\. GT=")
F=$(tr '\r' '\n' < "$LOG" | grep -c "Agent said=FAIL")
C=$(tr '\r' '\n' < "$LOG" | grep -c "Correct=True")

# birth time: macOS stat -f %B; Linux stat -c %W (0 if unsupported -> mtime)
if BIRTH=$(stat -f %B "$LOG" 2>/dev/null); then :; else
  BIRTH=$(stat -c %W "$LOG" 2>/dev/null || echo 0)
  [ "$BIRTH" -eq 0 ] && BIRTH=$(stat -c %Y "$LOG")
fi
NOW=$(date +%s)
E=$((NOW - BIRTH))

if [ "$L" -gt 0 ]; then
  PACE=$((E / L))
  ETA_MIN=$(((140 - L) * PACE / 60))
else
  PACE=0; ETA_MIN=0
fi

printf "%s  %s\n" "$ALIVE" "$LOG"
printf "loops=%s/140  fails=%s  correct=%s  elapsed=%dmin  pace=%ds/loop  eta=%dmin\n" \
  "$L" "$F" "$C" $((E / 60)) "$PACE" "$ETA_MIN"

if [ "$L" -gt 70 ]; then
  BF=$(tr '\r' '\n' < "$LOG" | grep "complete\. GT=" | head -70 | grep -c "Agent said=FAIL")
  BC=$(tr '\r' '\n' < "$LOG" | grep "complete\. GT=" | head -70 | grep -c "Correct=True")
  TF=$(tr '\r' '\n' < "$LOG" | grep "complete\. GT=" | tail -n +71 | grep -c "Agent said=FAIL")
  TC=$(tr '\r' '\n' < "$LOG" | grep "complete\. GT=" | tail -n +71 | grep -c "Correct=True")
  TN=$((L - 70))
  echo "baseline (70): correct=$BC fails=$BF | trained ($TN so far): correct=$TC fails=$TF"
fi

echo "--- last lines ---"
tr '\r' '\n' < "$LOG" | grep "complete\. GT=" | tail -3
tr '\r' '\n' < "$LOG" | grep -E "Baseline accuracy|Trained accuracy|Improvement" || true
