# MeetMind Agentic RL — Project Report

Three-stage pipeline: (1) supervised fine-tuning of a small probability predictor,
(2) agentic trace generation where persona agents use that predictor as a tool,
(3) Direct Preference Optimization of the agent model from its own traces, with a
baseline-vs-trained evaluation.

**Headline results**

| Stage | Deliverable | Result |
|---|---|---|
| 1 — SFT | `models/sft_prediction_model_gemma_270m/` | correlation 0.268, MAE 0.257, accuracy 54% (n=50) |
| 2 — Traces | `agent_traces_20260804_003816.csv` (70 traces) | 70/70 completed naturally, SFT tool used in 63/70 |
| 3 — DPO | `qwen3-dpo-adapter-v1/` | baseline 37.1% → trained 35.7% (−1.4 pp, n=70/model); FAIL rate 43% → 37% |

Training ran on an Udacity T4 workspace (after upgrading its stock image — see
Environment); trace generation and development ran on an Apple-Silicon Mac with
Ollama.

---

## Stage 1 — SFT probability predictor

**Task.** LoRA-fine-tune `google/gemma-3-270m-it` to emit
`{"probability": 0.XX, "reason": "word"}` for a two-attendee description plus a
time slot, trained on 2000 simulator-generated rows (1600 train / 400 val).

**Why a Causal LM rather than a classifier head?** The model *writes* the JSON
token by token, digits included, conditioned on everything before them. That
keeps the base model's instruction-following intact, which is what lets Stage 2
call the fine-tuned model as a drop-in text tool rather than wiring in a bespoke
regression head.

**Hyperparameters and why**

| Parameter | Value | Rationale |
|---|---|---|
| lora_r / alpha | 16 / 32 | r=8 underfit the probability distribution's tails; r=32 was unstable. alpha=2r convention. Only q_proj/v_proj adapted (<1% of params). |
| lora_dropout | 0.1 | 2000 rows repeated over epochs is enough to memorise. |
| learning_rate | 3e-5 | 5e-5 produced loss spikes; 1e-5 needed ~2× the epochs. |
| scheduler | cosine, restarted per epoch | Empirically load-bearing — see the experiment below. |
| epochs | **4** (early stop; config default 12) | Validation correlation peaks at epoch 3–4 and collapses afterward. |
| batch | 8 effective (8×1 on T4, 2×4 on MPS) | `--batch-size 8 --grad-accum 1` cuts forward passes 4× on the T4. |
| early_stopping_loss | 0.4 | The `reason` field is a ~280-way long tail of near-synonyms; a well-generalising model plateaus at loss 0.6–0.9. Below 0.4 means memorisation. |
| max_seq_length / val max_new_tokens | 512 / 64 | p99 input is ~310 tokens; the target JSON is ~20 tokens. |

**Finding 1: overfitting collapse.** A 12-epoch run peaked at correlation +0.37
(epoch 4) and then collapsed into a constant predictor (`0.230` for every input)
by epoch 6, finishing at correlation 0.002 while training loss kept improving
(1.57 → 1.36) — the model memorised reason-text while abandoning input
discrimination. Validation correlation by epoch: −0.28, —, +0.32, **+0.37**,
+0.30, then −0.2 to +0.1 noise. The fix: stop at 4 epochs.

**Finding 2: LR schedule experiment (restarts vs continuous).** The training
loop calls `trainer.train()` once per epoch, restarting a one-epoch cosine cycle
each time. We hypothesised the repeated returns to peak LR caused the collapse
and tested one continuous cosine decay across all epochs (8-epoch run).
Result: **worse** — final correlation −0.044 (constant predictor) vs 0.268 for
4 epochs with restarts. The per-epoch anneal-to-zero turns out to be
load-bearing: each epoch ends in a briefly "settled" model, and stopping after
epoch 4's anneal captures the best one. The change was reverted
(commits `3d4ba06` → `034630c`).

**Metric choice.** Correlation is the honest headline metric: a constant
predictor scores exactly 0 there, while MAE and threshold-accuracy can look
respectable on collapsed output. (Our hardened JSON parser can extract a
constant `0` from degenerate outputs, making parse-rate and accuracy actively
misleading during early training.)

**Noise ceiling.** The simulator's ground truth is deliberately stochastic:
`_calculate_meeting_success` computes a success probability, rolls
`random() < success_prob * 2`, then returns a partly *inverted* transform of the
underlying compatibility. The labels therefore cap achievable correlation well
below 1.0; iterating Stage 1 past ~0.3 correlation is chasing noise, so we
banked 0.268 (n=50, end-to-end through the same serving path Stage 2 uses) and
moved on.

**Starter bugs fixed along the way** (originals preserved in
`data_classes.ORIGINAL.py`): missing `import re` that silently produced
`nan` metrics; undefined `npcpy_get_llm_response`; train/serve prompt skew
between the CSV format and `MeetingPredictor.predict`; fragile
`split("model")` decoding; an N² epoch loop (`num_train_epochs=N` *and*
`range(N)` around `train()`); bf16 forced on the bf16-less T4; and a
non-greedy JSON regex that broke on fenced/nested/truncated model output.

---

## Stage 2 — Agent trace generation

**Task.** 7 persona agents (balanced, cautious, optimistic, data-driven,
intuitive, strategic, efficiency-focused) decide YES/NO on meeting scenarios
using 4 tools — three qwen3-backed analysis tools plus
`predict_meeting_success_tool`, which serves the Stage 1 adapter through a
process-wide cached `MeetingPredictor` (one disk load per run instead of one
per tool call).

**Key settings.** `max_iterations=8` (room for all four tools plus reasoning
and the final JSON); trace agent `qwen3:0.6b` — deliberately matching Stage 3's
DPO target `Qwen/Qwen3-0.6B` so the preference data is approximately
**on-policy**; DPO's signal weakens when chosen/rejected completions come from a
different (larger) model than the one being trained.

**Finding: prompt staging is what makes a 0.6B agent work.** Three failed
smoke runs, one lesson each:

1. *Output schema injected mid-loop (starter behavior):* agents answered before
   seeing it, with near-miss keys (`final_recommendation`) — 0/7 completed.
2. *Schema in the system prompt from the start:* tool calling **fully
   suppressed** — agents "answered" at iteration 1 with fabricated tool results
   ("98% chance of success") and zero real tool calls.
3. *Staged prompting (final):* no JSON schema mentioned anywhere until at least
   one real tool call has happened; evidence-free answers are rejected with a
   corrective prompt; alias-tolerant parsing maps residual key-name misses.

**Full run health (70 traces, 10 per persona):** 70/70 completed naturally;
`predict_meeting_success_tool` called in 63/70; rewards 33 × 1.0 (correct) and
37 × 0.1 (incorrect, well-formed) — 47% agent accuracy against a ~50% base
rate; per-persona mean rewards genuinely differentiate (0.37–0.82). The CSV
saves `completed_naturally` and a plain `agent_recommendation` column — the
starter omitted the former even though Stage 3's reward function reads it,
which would have made honest reward recomputation impossible.

**Known skew, carried into Stage 3's design:** the agents said YES 87% of the
time, and *every* correct trace is a YES — there are zero correct NOs in the
dataset. All 9 NO answers were wrong.

---

## Stage 3 — DPO alignment and evaluation

**Reward design** (`calculate_reward`):

| Outcome | Reward | Note |
|---|---|---|
| Correct recommendation | **+1.0** | |
| Incorrect recommendation | **−0.5** | Negative, unlike Stage 2's 0.1 shaping value — the sign encodes chosen-vs-rejected for pairing. |
| Invalid format (not YES/NO) | −0.25 | |
| No tools used | −0.5 | |
| Did not complete naturally | −0.75 | |
| No parseable recommendation | **−1.0** | Reserved: the pairing filter drops reward ≤ −1.0, and a trace with no recommendation text can't serve even as a rejected sample. Returning −1.0 for *incorrect* answers (a tempting choice) would silently drop every rejected candidate and yield zero pairs. |

Rewards are **recomputed from raw CSV columns** (recommendation, tools_used,
completed_naturally, ground truth) rather than read back from the stored reward
column — that is why Stage 2 persists those columns.

**Pairing strategy.** Chosen = correct traces (+1.0); rejected = anything with
a reward gap ≥ 0.5 below (incorrect at −0.5 gives a 1.5 gap). Because every
correct trace says YES, unrestricted pairing would teach "always say YES"; two
mitigations: (a) most pairs contrast YES-right vs YES-wrong — same surface
recommendation, different reasoning quality — and (b) per-trace usage caps
(each chosen and each rejected trace used ≤ 2×) stop any single trace from
dominating. Result: **66 pairs (49 rejected-YES / 17 rejected-NO)**, seeded and
reproducible. Known simplification: each pair uses the chosen trace's prompt,
so the rejected completion answered a different scenario — trace-level DPO
rather than same-prompt sampling.

**DPO hyperparameters**

| Parameter | Value | Rationale |
|---|---|---|
| LoRA r / alpha / dropout | 8 / 16 / 0.05 | ~66 short JSON pairs: r=16 has nothing extra to learn; DPO's beta term already regularises, so light dropout. |
| target_modules | q, k, v projections | Preference learning mostly reshapes attention; k_proj added vs Stage 1's q/v. |
| learning_rate | 5e-6 | An order below SFT — DPO moves logits, not knowledge; 5e-5-class rates visibly collapse outputs on sets this small. |
| max_steps / batch | 100 / 2 effective | ~3 passes over 66 pairs: enough to move preferences, not memorise them. |
| beta | 0.1 | Standard data-vs-reference trade-off. |
| max_length / max_prompt_length | 1024 / 768 | Real data is ~400-token prompts + ~200-token completions; the starter's 8192 was pure padding waste. |

Training signals: loss 0.693 → ~0.63 over 100 steps, `rewards/accuracies`
reaching 1.0 on most late batches, positive and growing margins (to ~0.4) with
small absolute drift from the reference — preference movement without output
collapse.

**The adapter-directory trap.** The evaluation loads models with
`NPC(model=<path>, provider="transformers")`, which cannot resolve a bare LoRA
adapter directory. After training, the adapter is merged into the base and
saved as a full-weights directory (`qwen3-dpo-adapter-v1-merged/`), and the
evaluation is pointed at that; the adapter itself ships in
`qwen3-dpo-adapter-v1/`.

**Evaluation.** Fresh scenarios (seeded RNG, `ConferenceSimulator` with 2000
attendees), 7 personas × 10 scenarios × 2 models = 140 tool loops, run on the
Mac (the T4's wall-clock advantage disappears for this workload — the time is
dominated by reasoning-loop length and per-iteration model reloads, not raw
GPU speed — and the workspace idle-timeout cannot survive an ~11-hour run).

| Model | Accuracy | Loop-completion FAILs |
|---|---|---|
| Baseline `Qwen/Qwen3-0.6B` | 37.1% (26/70) | 30/70 (43%) |
| DPO-trained | 35.7% (25/70) | 26/70 (37%) |
| Improvement | **−1.4 pp** | **−6 pp FAIL rate** |

**Interpretation — an honest null result on accuracy.** The 1.4-point gap is
one answer out of seventy: statistically indistinguishable from zero. Two
real observations survive the noise:

1. *DPO moved the policy in exactly the direction it was trained.* The
   preference data consisted of well-formed final-JSON completions, and the
   trained model completed the tool loop more often (FAIL rate 43% → 37%).
   Training-time margins (`rewards/accuracies` → 1.0, growing positive
   margins) confirm genuine preference learning rather than a no-op.
2. *Better format compliance did not become better decisions.* The extra
   completed loops converted to wrong answers as often as right ones —
   unsurprising, since the preference pairs could not teach correctness
   directly: the dataset contained zero correct-NO examples, and each pair's
   rejected completion answered a different scenario than its prompt
   (trace-level DPO), so the learnable signal was format and style, not
   scenario-conditional judgment.

Both absolute accuracies are dragged well below the trace-generation runs'
~47–62% by the FAIL rate: under the `transformers` provider both models
fumble tool-call arguments far more than the same weights served by Ollama
did in Stage 2 — an infrastructure effect that hits baseline and trained
equally, so the comparison remains fair even though the absolute numbers are
depressed.

**What would likely make the delta positive** (future work): preference pairs
that share a prompt (sample multiple completions per scenario instead of
pairing across scenarios); correct-NO examples in the chosen set (more traces,
or persona prompts that elicit more NO decisions); and including tool-calling
turns in the preference data so DPO reinforces the *process*, not only the
final answer.

**Off-policy note.** Traces were generated with `qwen3:0.6b` (Ollama) and DPO
trained `Qwen/Qwen3-0.6B` (transformers) — the same weights served by different
runtimes/quantizations, so the data is near-on-policy by construction; the
alternative (generating with the stronger `qwen3:1.7b`, as the starter did)
was rejected for exactly this reason.

---

## Environment

Two machines, two library generations, one codebase: the Udacity workspace
image ships transformers 4.46.3 (predates Gemma 3 support) and TRL 0.11, while
the Mac runs transformers 5.x / TRL 1.x. `compat.py` introspects the installed
trainer/config classes and passes only accepted arguments, so identical source
runs on both. `setup_workspace.sh` upgrades the workspace image (CUDA torch,
current transformers/TRL) — the stock image cannot load the Gemma 3 base model
at all. The T4 has no bf16 (sm_75); `compat.resolve_precision` selects fp16
there and full fp32 where mixed precision is unsupported. GPU speedup was
~5.5× over MPS for SFT; Ollama was installed manually in the (systemd-less)
workspace container for Stages 2–3.

## Artifacts

- `models/sft_prediction_model_gemma_270m/` — Stage 1 LoRA adapter +
  `training_run.json` + `final_validation.json`
- `agent_traces_20260804_003816.csv` — 70 traces (plus the 21-trace smoke CSV)
- `qwen3-dpo-adapter-v1/` — Stage 3 DPO LoRA adapter
- `dpo_evaluation_results.json` — per-scenario evaluation record
