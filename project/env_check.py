#!/usr/bin/env python3
"""
MeetMind project — environment preflight.

Run this from the `project/` directory on any machine you intend to train on:

    python env_check.py

It answers, in order: can this box load Gemma 3 at all, which TRL API does it
speak, is Ollama reachable with the right models, and is the dataset intact.
Everything is a soft check -- nothing here trains or downloads a model.
"""

import importlib
import json
import os
import sys
import urllib.error
import urllib.request

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(status, label, detail=""):
    results.append((status, label, detail))
    mark = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {label}" + (f"\n         {detail}" if detail else ""))


def version_of(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


# --------------------------------------------------------------------------
print("\n=== Python & core libraries ===")

record(
    PASS if sys.version_info >= (3, 10) else WARN,
    f"Python {sys.version.split()[0]}",
    "" if sys.version_info >= (3, 10) else "project targets 3.10+",
)

for pkg in ["torch", "transformers", "peft", "trl", "datasets", "pandas", "numpy", "npcpy"]:
    v = version_of(pkg)
    record(PASS if v else FAIL, f"{pkg} {v or 'NOT INSTALLED'}")

# --------------------------------------------------------------------------
print("\n=== Compute device ===")

try:
    import torch

    if torch.cuda.is_available():
        record(PASS, f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        record(PASS, "Apple MPS available")
        device = "mps"
    else:
        record(
            WARN,
            "CPU only",
            "Gemma-270m SFT on 2000 examples will take hours. Expect ~30-60 min/epoch.",
        )
        device = "cpu"
    record(PASS, f"torch build: {torch.__version__}")
except Exception as e:
    record(FAIL, "torch unusable", str(e)[:200])
    device = None

# --------------------------------------------------------------------------
print("\n=== Gemma 3 architecture support ===")
# gemma-3-270m-it uses model_type 'gemma3_text', which transformers only learned
# in 4.50.0. On older transformers the load fails with an unrecognised-arch error
# no amount of hyperparameter tuning will fix.

try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    gemmas = sorted(k for k in CONFIG_MAPPING_NAMES if "gemma" in k)
    if any("gemma3" in g for g in gemmas):
        record(PASS, "transformers knows gemma3", f"registered: {gemmas}")
    else:
        record(
            FAIL,
            "transformers CANNOT load google/gemma-3-270m-it",
            f"only {gemmas} registered. Need transformers>=4.50 (4.55+ for the 270m). "
            "Either upgrade here or run Stage 1 on a machine that has it.",
        )
except Exception as e:
    record(WARN, "could not introspect transformers config map", str(e)[:200])

# --------------------------------------------------------------------------
print("\n=== TRL API generation ===")
# trl <=0.11 accepts TrainingArguments and takes `tokenizer=`;
# trl >=0.12 wants SFTConfig/DPOConfig and `processing_class=`.

try:
    import inspect

    from trl import DPOTrainer, SFTTrainer

    sft_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    dpo_params = set(inspect.signature(DPOTrainer.__init__).parameters)
    api = "new (processing_class)" if "processing_class" in dpo_params else "old (tokenizer)"
    record(PASS, f"TRL API: {api}")
    record(
        PASS,
        "trainer kwargs",
        f"SFTTrainer accepts: {sorted(sft_params - {'self'})[:8]}...\n"
        f"         DPOTrainer accepts: {sorted(dpo_params - {'self'})[:8]}...",
    )
except Exception as e:
    record(FAIL, "trl unusable", str(e)[:200])

# --------------------------------------------------------------------------
print("\n=== Hugging Face auth (Gemma is a gated repo) ===")

try:
    from huggingface_hub import HfApi

    try:
        who = HfApi().whoami()
        record(PASS, f"HF logged in as {who.get('name')}")
    except Exception:
        record(
            FAIL,
            "not logged in to Hugging Face",
            "run: huggingface-cli login   (and accept the license at "
            "https://huggingface.co/google/gemma-3-270m-it)",
        )
except Exception:
    record(WARN, "huggingface_hub not importable — cannot check auth")

# --------------------------------------------------------------------------
print("\n=== Ollama (needed for Stages 2 and 3) ===")

REQUIRED_OLLAMA = ["qwen3:0.6b", "qwen3:1.7b"]
try:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
        tags = json.loads(r.read())
    have = {m["name"] for m in tags.get("models", [])}
    record(PASS, f"Ollama server reachable ({len(have)} models)")
    for m in REQUIRED_OLLAMA:
        if m in have or f"{m}-instruct" in have:
            record(PASS, f"model present: {m}")
        else:
            record(FAIL, f"model missing: {m}", f"run: ollama pull {m}")
    extras = sorted(have - set(REQUIRED_OLLAMA))
    if extras:
        record(PASS, "other local models", ", ".join(extras))
except (urllib.error.URLError, OSError) as e:
    record(
        WARN,
        "Ollama not reachable on localhost:11434",
        f"{type(e).__name__}. Fine if this box only runs Stage 1; required for Stages 2-3.",
    )

# --------------------------------------------------------------------------
print("\n=== Project files ===")

csv_path = "data/sft_training_data.csv"
if os.path.exists(csv_path):
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        need = {"input_text", "target_probability", "target_reason", "ground_truth_prob"}
        missing = need - set(df.columns)
        if missing:
            record(FAIL, "CSV missing columns", str(missing))
        else:
            yes = (df["ground_truth_prob"] > 0.5).mean()
            record(
                PASS,
                f"dataset: {len(df)} rows",
                f"label balance: {yes:.1%} YES / {1 - yes:.1%} NO",
            )
    except Exception as e:
        record(FAIL, "CSV unreadable", str(e)[:200])
else:
    record(FAIL, f"missing {csv_path}")

# --------------------------------------------------------------------------
print("\n=== data_classes.py health ===")

try:
    import data_classes

    if hasattr(data_classes, "re"):
        record(PASS, "data_classes imports `re`", "safe_extract_json will not NameError")
    else:
        record(
            FAIL,
            "data_classes is missing `import re`",
            "safe_extract_json raises NameError on every call. In starter_sft.py this is "
            "swallowed by a bare except, so validation silently returns nan.",
        )

    if hasattr(data_classes, "npcpy_get_llm_response"):
        record(PASS, "npcpy_get_llm_response is defined")
    else:
        record(
            FAIL,
            "npcpy_get_llm_response is undefined",
            "PersonDescriptor.generate_description will NameError -> Stage 2 dies on "
            "the first scenario.",
        )

    # exercise the two broken paths for real
    try:
        data_classes.safe_extract_json('<start_of_turn>model\n{"probability": 0.5}')
        record(PASS, "safe_extract_json round-trips")
    except NameError as e:
        record(FAIL, "safe_extract_json NameError", str(e)[:120])
    except Exception as e:
        record(WARN, f"safe_extract_json raised {type(e).__name__}", str(e)[:120])

except Exception as e:
    record(FAIL, "cannot import data_classes", f"{type(e).__name__}: {str(e)[:200]}")

# --------------------------------------------------------------------------
print("\n" + "=" * 66)
fails = [r for r in results if r[0] == FAIL]
warns = [r for r in results if r[0] == WARN]
print(f"{len(results) - len(fails) - len(warns)} passed, {len(warns)} warnings, {len(fails)} failures")
if fails:
    print("\nBlocking issues:")
    for _, label, detail in fails:
        print(f"  - {label}")
print("=" * 66)
sys.exit(1 if fails else 0)
