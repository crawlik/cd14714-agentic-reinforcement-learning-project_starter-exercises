# Running this project

Two machines, different jobs:

| | Udacity Cloud Workspace | local Mac |
|---|---|---|
| hardware | T4 GPU (once the image is fixed) | Apple Silicon / MPS |
| stock torch | `2.8.0+cpu` — **GPU unusable** | `2.12.1` with MPS |
| stock transformers | `4.46.3` — **cannot load Gemma 3** | `5.13.0` — fine |
| stock trl | `0.11.4` (old API) | `1.7.1` (new API) |
| Ollama | not available | required for Stages 2–3 |
| runs | Stage 1 (SFT) | Stages 2–3 (traces, DPO, eval) |

`compat.py` bridges the TRL API difference, so the same source files run on both.

---

## One-time setup

### Udacity Cloud Workspace

The lab image needs repairing before Stage 1 will run at all — `transformers 4.46.3`
predates Gemma 3, so `google/gemma-3-270m-it` fails to load regardless of hyperparameters.

```bash
cd ~/project                 # wherever you cloned the repo
bash setup_workspace.sh      # CUDA torch + transformers>=4.56 + matching trl/peft
huggingface-cli login        # Gemma is gated — accept the license on the model page first
python env_check.py
```

`setup_workspace.sh` also installs a CUDA build of torch. That alone may unblock you
without waiting on ticket 2200583 — the driver is present (`nvidia-smi` sees the T4),
the lab just shipped a CPU-only wheel. Workspace sessions are ephemeral, so re-run the
script after the box is recycled.

### Mac

```bash
source agent-course-env/bin/activate
cd cd14714-agentic-reinforcement-learning-project_starter-exercises/project
huggingface-cli login
ollama pull qwen3:0.6b
ollama pull qwen3:1.7b
python env_check.py
```

`env_check.py` exits non-zero if anything blocking is wrong. Read its output before
running anything else — it catches the failure modes that otherwise look like
"my model didn't learn."

---

## Stage 1 — SFT probability predictor

```bash
cd project

# pipeline check: 1 epoch on 80 rows, writes to models/_smoke_test_adapter
python starter_sft.py --smoke

# the real run
python starter_sft.py
```

Expect roughly:

- **T4** ≈ 2–4 min/epoch → ~40 min for 12 epochs
- **MPS** ≈ 8–15 min/epoch → 2–3 hours
- **CPU only** ≈ 45–90 min/epoch → don't

Useful flags: `--epochs N`, `--train-size N`, `--output DIR`, `--skip-final-check`.

**Output:**

- `models/sft_prediction_model_gemma_270m/` — the adapter (a submission deliverable)
- `models/sft_prediction_model_gemma_270m/training_run.json` — hyperparameters plus
  per-epoch loss / correlation / MAE / accuracy, ready to paste into the report
- `models/sft_prediction_model_gemma_270m/final_validation.json` — metrics from
  reloading the saved adapter through `MeetingPredictor`, i.e. the exact path
  Stage 2's tool uses

**Reading the output.** Three numbers per epoch:

- **correlation** — how well the predicted probability tracks the simulator's.
  Do not expect this near 1.0. `ConferenceSimulator._calculate_meeting_success`
  rolls `random() < success_prob * 2` and then returns `1 - success_prob` on
  success and `success_prob` on failure, so the label is a stochastic, partly
  inverted transform of the underlying compatibility. There is irreducible noise
  in the target by design.
- **MAE** — mean absolute error on the probability. Lower is better; a constant
  predictor that always says 0.47 would score about 0.24.
- **accuracy** — fraction landing on the correct side of 0.5. This is the >50%
  the rubric grades. Chance is ~50% (labels are 47.6% YES).

If training loss falls below `early_stopping_loss = 0.4` the run stops: the reason
field is a ~280-way long tail of near-synonyms, so a well-generalising model
plateaus around 0.6–0.9. Under 0.4 means it is memorising descriptions.

**Then copy the adapter to the Mac** — it is only a few MB:

```bash
# from the Mac, or just commit it and pull
scp -r workspace:~/project/models/sft_prediction_model_gemma_270m models/
```

---

## Stages 2 and 3

Not written yet — next step once Stage 1 produces real numbers.

---

## Git flow

Nothing here is gitignored except `.DS_Store` and `CLAUDE.md`, so the model
directories will be committed, which is what you want: the graded submission needs
`models/sft_prediction_model_gemma_270m/` and `qwen3-dpo-adapter-v1/` in full.

```bash
git add project/
git commit -m "Stage 1: complete SFT implementation + environment fixes"
git push origin main
```

Then in the workspace:

```bash
git pull origin main
bash setup_workspace.sh
```

Watch the repo size once adapters land — a LoRA adapter on q_proj/v_proj of a
270M model is small (single-digit MB), but do not commit any full merged model.

---

## What changed from the starter files

| file | change |
|---|---|
| `data_classes.py` | added missing `import re` (`safe_extract_json` raised `NameError` on every call — swallowed by a bare `except` in the SFT validation loop, so validation silently returned `nan`) |
| | defined `npcpy_get_llm_response`, which `PersonDescriptor.generate_description` called but nothing ever provided — Stage 2 died on scenario one |
| | added `PREDICTION_PROMPT_TEMPLATE` / `build_prediction_prompt` so training and serving use one format. `MeetingPredictor` previously built a `". "`-joined prompt while the CSV is newline-joined with a blank line before the question — train/serve skew on every Stage 2 tool call |
| | `MeetingPredictor.predict` now decodes greedily, reads only the generated tokens, and parses via `safe_extract_json` instead of `result.split("model")[-1]` (which broke whenever a description contained the word "model") |
| | added `get_meeting_predictor()`, a process-wide cache — the Stage 2 tool would otherwise reload Gemma from disk on every single tool call |
| | original preserved as `data_classes.ORIGINAL.py` |
| `starter_sft.py` | all `YOUR CODE HERE` filled; `validate_model` implemented; fixed the N² epoch loop (the starter set `num_train_epochs=N` *and* wrapped `train()` in `range(N)`) |
| | LR scheduler rebuilt per epoch — otherwise the cosine schedule is exhausted after epoch 1 and the LR sits at ~0 |
| | `bf16` no longer forced on for all CUDA devices; the T4 is sm_75 and has no bf16 support |
| | added directional YES/NO accuracy, `--smoke` mode, and JSON metric dumps for the report |
| `compat.py` | new — introspects the installed TRL/transformers and passes only arguments they accept, so one file runs on both the classroom image and the Mac |
| `env_check.py` | new — preflight for every failure mode above |
| `setup_workspace.sh` | new — repairs the lab image (CUDA torch + Gemma-3-capable transformers) |
