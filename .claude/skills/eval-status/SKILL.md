---
name: eval-status
description: Report the status of the running (or finished) Stage 3 DPO evaluation on this machine — loops completed, FAIL count, accuracy so far, pace, and ETA.
---

# eval-status

Report the current state of the Stage 3 DPO evaluation (`starter_agentic_rlft.py --skip-training --eval-scenarios N`) from its log file. Works while the run is in flight and after it finishes.

## Preferred: run the bundled script

Run exactly:

    bash .claude/skills/eval-status/eval_status.sh

(from the repo root; it is pre-approved in the permission allowlist). It prints liveness, loop counts, FAIL/correct counts (split baseline vs trained past loop 70), pace, ETA, the last completion lines, and the final accuracy block when the run has finished. Then summarize per step 6 below.

## Fallback: manual steps (if the script is missing)

1. Locate the newest eval log: `ls -t project/logs/dpo_eval_*.log | head -1` (fall back to any `eval*.log` in `project/`). All subsequent commands run from the repo root; adjust paths if the session cwd is already `project/`.

2. Check liveness: `pgrep -f starter_agentic_rlft` — report RUNNING or NOT RUNNING.

3. Gather stats from the log (strip carriage returns with `tr '\r' '\n'` first — progress bars pollute the lines):
   - loops completed: count of `complete. GT=` lines (total expected = scenarios × 7 personas × 2 models; 140 for `--eval-scenarios 10`)
   - FAIL count: count of `Agent said=FAIL`
   - correct count: count of `Correct=True`
   - last 2–3 completion lines for context (which persona/model phase it is in: `_p0`..`_p6` suffix = persona index; the first half of loops is the baseline model, the second half the DPO-trained model)

4. Compute pace and ETA:
   - elapsed = now − log file birth time (`stat -f %B <log>` on macOS gives birth as epoch seconds; on Linux use `stat -c %W`, falling back to `%Y` of the oldest line if birth is 0)
   - pace = elapsed / loops; ETA = remaining loops × pace
   - If the run is past loop 70 (of 140), split FAIL and correct counts by half: loops 1–70 = baseline, 71–140 = trained. Report the two models separately — per-model FAIL rate is a first-class metric here, not just accuracy.

5. If the run has FINISHED (process gone and the log contains `Baseline accuracy`): report the final block verbatim — baseline accuracy, trained accuracy, improvement — and note that `dpo_evaluation_results.json` holds the per-scenario record.

6. Report in 2–5 lines: liveness, loops/total, per-model or overall FAIL rate, accuracy so far, pace, ETA (wall-clock time of day, not just duration). Flag explicitly if: the process died before completing all loops, pace exceeds 600 s/loop, or the trained-model phase shows a FAIL rate above ~35%.
