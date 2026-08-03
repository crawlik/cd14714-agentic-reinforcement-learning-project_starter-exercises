# MeetMind Agentic RL — Project Plan

**Split:** Stage 1 (SFT) in the Udacity Cloud Workspace · Stages 2–3 (traces, DPO, eval) on the Mac
**Mode:** stage by stage — each stage runs and produces real numbers before we start the next

---

## Things that will bite before any code is written

These came out of reading the starter files. Fixing them up front is Phase 0.

### 1. `data_classes.py` has two crash bugs

**`safe_extract_json` uses `re` but `re` is never imported.** Every call raises `NameError`. In `starter_sft.py` the validation loop wraps it in `try/except Exception: continue`, so the failure is *silent* — you get empty prediction arrays and `nan` correlation/MAE, and it looks like your model didn't learn. This one costs people a whole evening.

**`PersonDescriptor.generate_description` calls `npcpy_get_llm_response`, which is never imported or defined.** Stage 2 crashes on the first scenario. Fix: `from npcpy.llm_funcs import get_llm_response as npcpy_get_llm_response` (and pass a model/provider — it currently passes only `temperature`).

### 2. Ollama models aren't pulled

Your local models are `llama3.2`, `gemma4`, `deepseek-r1:14b`. The project wants:

```bash
ollama pull qwen3:0.6b   # tool implementations
ollama pull qwen3:1.7b   # trace-generating agent
```

`llama3.2:latest` is a usable tool-calling fallback if qwen3 misbehaves; `deepseek-r1:14b` is too slow for a 7-persona × N-scenario loop.

### 3. Library version skew on the Mac

`agent-course-env` has **transformers 5.13.0, trl 1.7.1, peft 0.19.1, datasets 5.0.0**. The starter code targets the trl ~0.8 era:

- `SFTTrainer(model=…, args=TrainingArguments(…), peft_config=…)` — TRL 1.x wants `SFTConfig`, not `TrainingArguments`
- `DPOTrainer(model, args=…, train_dataset=…, peft_config=…)` — TRL 1.x needs `processing_class=tokenizer`
- the starter defines its own `SFTConfig` dataclass, which **name-collides** with `trl.SFTConfig`

First thing to check in the Udacity workspace: `pip list | grep -E 'trl|peft|transformers|datasets'`. If the classroom has the old pinned versions, Stage 1 code written for it won't run as-is on your Mac. We either pin the Mac to match, or write version-tolerant calls. Decide once, early.

### 4. Gemma is a gated HF repo

`google/gemma-3-270m-it` needs an accepted license + `huggingface-cli login` on **both** machines — the Mac needs the base model too, since Stage 2 loads the adapter on top of it.

### 5. Two traps on the rubric path

**The reward-gap filter will eat your rejected samples.** `create_preference_dataset_from_traces` does `valid_df[valid_df['reward'] > -1.0]`. If `calculate_reward` returns `-1.0` for *incorrect* recommendations, every rejected candidate is dropped and you get zero pairs. Reserve `-1.0` for malformed/no-recommendation; use `-0.5` for incorrect (which is also what the project instructions say). Gap becomes `1.0 − (−0.5) = 1.5`, comfortably over the required 0.5.

**The trace CSV doesn't save `completed_naturally`.** The rubric wants Stage 3 to *apply* `calculate_reward` to the loaded traces, but that function reads `completed_naturally` — which `AgentTraceCollector.save_traces_to_file` never writes. We add the column in Stage 2 so Stage 3 can honestly recompute rewards instead of just reading the `reward` column back.

### 6. Two performance landmines

**`predict_meeting_success_tool` would reload the 270M model from disk on every tool call.** Across 7 personas × N scenarios × up to 8 iterations that's hundreds of model loads. Cache it in a module-level singleton.

**The final eval is bigger than it looks.** `run_local_agent_evaluation` loops all 7 personas × N scenarios × 2 models × 8 iterations, and each scenario also builds a `ConferenceSimulator(num_attendees=2000)` and makes 2 description-generating LLM calls. At the default 3 scenarios that's 42 agent loops. Plan to trim personas for eval, or budget the time.

### 7. `starter_sft.py` trains N² epochs

`TrainingArguments(num_train_epochs=N)` already runs N epochs — and `__main__` wraps `trainer.train()` in `for epoch in range(N)`. With N=15 that's 225 epochs of a 8k-row dataset. Restructure to 1 epoch per trainer call inside the loop, so the per-epoch validation and early stopping actually mean something.

### 8. Design point worth a paragraph in your report

Traces are generated with `qwen3:1.7b` (Ollama) but DPO trains `Qwen/Qwen3-0.6B` (transformers). DPO assumes preference data drawn from roughly the policy you're training — off-policy data from a different, larger model weakens the signal. Recommendation: generate traces with **qwen3:0.6b** to match. Mentioning the tradeoff explicitly is free rubric points under "design choices."

---

## Phases

### Phase 0 — Environment + bug fixes (~1 h, both machines)

- `check_env.py`: prints versions, MPS/CUDA availability, Ollama model list, HF auth status, whether the CSV and dirs exist. Green/red, no guessing.
- Patch `data_classes.py`: `import re`, wire up `npcpy_get_llm_response`.
- `ollama pull qwen3:0.6b qwen3:1.7b`; `huggingface-cli login` both places.
- Record the Udacity workspace's library versions → decide the pinning strategy.

**Done when:** `check_env.py` is all green on both machines.

### Phase 1 — SFT predictor (Udacity workspace, GPU)

- Fill `SFTConfig`: `lora_r=8`, `lora_alpha=16`, `lora_dropout=0.1`, `num_train_epochs=15`, `learning_rate=2e-5`, `weight_decay=0.01` as the starting point — then tune against real validation numbers.
- Implement `validate_model` (the rubric checks it even though `__main__` uses the inline loop).
- Early-stop threshold `0.4`; `max_new_tokens=64` (enough for `{"probability": 0.XX, "reason": "word"}` plus slack, small enough to stay fast).
- Fix the N² epoch loop.
- Train, read the correlation/MAE, adjust, retrain. **This is the iteration-heavy stage.**

**Deliverable:** `models/sft_prediction_model_gemma_270m/` + a table of hyperparameters → correlation/MAE for the report. Download the adapter (small — LoRA on q_proj/v_proj of a 270M model) to the Mac.

### Phase 2 — Agent traces (Mac, Ollama)

- Implement `predict_meeting_success_tool` with the cached `MeetingPredictor`.
- `max_iterations=8`, `num_traces_per_agent=3` for a smoke test, then **10** (7 personas × 10 = 70 traces; the README asks for 50+).
- Add `completed_naturally` to the CSV writer.
- Sanity-check the output: are 2–5 tools being called per trace? Is the reward distribution actually varied, or is everything -0.75 because the model never emits clean JSON?

**Deliverable:** `agent_traces_YYYYMMDD_HHMMSS.csv` with real spread in the reward column.

### Phase 3 — DPO + evaluation (Mac)

- `calculate_reward`: `+1.0` correct · `−0.5` incorrect · `−0.25` invalid format · `−0.5` no tools · `−0.75` didn't complete · `−1.0` no parseable recommendation.
- Pairing: recompute rewards from the CSV, require gap `≥ 0.5`, cap pairs per chosen-trace so a handful of good traces don't dominate, and keep YES/NO roughly balanced.
- LoRA `r=8, alpha=16, dropout=0.05`, targets `["q_proj","k_proj","v_proj"]`. DPO: `lr=5e-6`, `max_steps=100`, `beta=0.1`, `weight_decay=0.01`.
- **Merge the adapter into the base and save a merged dir for eval** — `NPC(model="./qwen3-dpo-adapter-v1", provider="transformers")` won't load a bare adapter directory. `load_trained_model` already does the merge.
- Run baseline vs trained.

**Deliverable:** `qwen3-dpo-adapter-v1/` + baseline %, trained %, improvement.

### Phase 4 — Report + submission (~2 h)

Report covers, per stage: hyperparameters and why, validation correlation/MAE, trace counts and tool-usage patterns, reward design rationale (including the `> -1.0` filter interaction), pairing strategy, DPO hyperparameters, final accuracy comparison. Plus the "why is this a Causal LM task?" question the starter plants at line 115.

Zip as `LastName_FirstName_Project.zip` with both model directories intact.

---

## Realistic time

| Phase | Wall clock | Your attention |
|---|---|---|
| 0 — env + fixes | ~1 h | high |
| 1 — SFT (with retries) | 3–6 h | medium (waiting on training) |
| 2 — traces | 2–4 h | low (long unattended run) |
| 3 — DPO + eval | 2–4 h | medium |
| 4 — report | ~2 h | high |

The failure mode isn't difficulty — it's Stage 2 running for three hours and producing 70 traces that all have reward −0.75 because the agent never emitted parseable JSON. Hence the 3-trace smoke test before the real run.

---

## To start Phase 0 I need

1. `pip list` from the Udacity Cloud Workspace (transformers / trl / peft / datasets versions)
2. Whether that workspace has a GPU and an HF token already configured
3. Your last name, for the submission zip
