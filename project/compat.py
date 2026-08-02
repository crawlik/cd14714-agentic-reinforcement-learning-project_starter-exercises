"""
Version-compatibility shims for TRL / transformers.

The two machines this project runs on speak different TRL dialects:

    Udacity Cloud Workspace : transformers 4.46.3, trl 0.11.4, peft 0.13.2
    local Mac (MPS)         : transformers 5.13.0, trl 1.7.1,  peft 0.19.1

The differences that actually break the starter code:

  * trl <= 0.11 lets you hand ``TrainingArguments`` to ``SFTTrainer``;
    trl >= 0.12 requires an ``SFTConfig`` / ``DPOConfig``.
  * the tokenizer argument was renamed ``tokenizer=`` -> ``processing_class=``.
  * ``SFTConfig.max_seq_length`` was renamed to ``max_length``.
  * ``dataset_text_field`` / ``max_seq_length`` used to be trainer kwargs and
    are now config fields.

Rather than pinning one environment, everything below introspects the installed
classes and passes only the arguments they actually accept. The same source file
therefore runs on either box, which matters because the code is graded on the
classroom image but developed locally.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Dict

import torch


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(device: str) -> Dict[str, bool]:
    """Pick fp16/bf16 for the device we actually landed on.

    Notably: the T4 in the Udacity lab is sm_75 and has no bf16 support, so the
    starter's ``bf16 = True`` on any CUDA device is wrong there. fp16 is also
    unsupported for mixed-precision training on MPS.
    """
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


# ---------------------------------------------------------------------------
# kwarg filtering
# ---------------------------------------------------------------------------
def _accepted_fields(cls) -> set:
    """Field names a (dataclass) config class will accept."""
    try:
        return {f.name for f in dataclasses.fields(cls)}
    except TypeError:
        return set(inspect.signature(cls.__init__).parameters) - {"self"}


def _accepted_params(fn) -> set:
    return set(inspect.signature(fn).parameters) - {"self"}


def _build_config(cls, requested: Dict[str, Any], aliases: Dict[str, str]) -> Any:
    """Instantiate ``cls`` with whatever subset of ``requested`` it understands.

    ``aliases`` maps our canonical name -> alternative name to try if the
    canonical one is not a field (e.g. max_length -> max_seq_length).
    """
    fields = _accepted_fields(cls)
    kwargs, dropped = {}, []
    for key, value in requested.items():
        if key in fields:
            kwargs[key] = value
        elif key in aliases and aliases[key] in fields:
            kwargs[aliases[key]] = value
        else:
            dropped.append(key)
    if dropped:
        print(f"[compat] {cls.__name__} ignores: {', '.join(sorted(dropped))}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------
def make_sft_training_args(**requested):
    """Return the right args object for the installed TRL."""
    try:
        from trl import SFTConfig as TRLSFTConfig  # trl >= 0.9

        return _build_config(
            TRLSFTConfig,
            requested,
            aliases={"max_length": "max_seq_length", "max_seq_length": "max_length"},
        )
    except ImportError:
        from transformers import TrainingArguments

        # TrainingArguments knows nothing about text fields / seq length
        for k in ("dataset_text_field", "max_length", "max_seq_length", "packing"):
            requested.pop(k, None)
        return _build_config(TrainingArguments, requested, aliases={})


def make_sft_trainer(model, tokenizer, train_dataset, peft_config, args, **extra):
    from trl import SFTTrainer

    params = _accepted_params(SFTTrainer.__init__)
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    # older TRL wants these on the trainer rather than the config
    for key, value in extra.items():
        if key in params:
            kwargs[key] = value

    return SFTTrainer(**kwargs)


# ---------------------------------------------------------------------------
# DPO  (used by Stage 3)
# ---------------------------------------------------------------------------
def make_dpo_training_args(**requested):
    from trl import DPOConfig

    return _build_config(
        DPOConfig,
        requested,
        aliases={"max_length": "max_seq_length", "max_completion_length": "max_target_length"},
    )


def make_dpo_trainer(model, tokenizer, train_dataset, peft_config, args, ref_model=None):
    from trl import DPOTrainer

    params = _accepted_params(DPOTrainer.__init__)
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if "ref_model" in params:
        kwargs["ref_model"] = ref_model
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return DPOTrainer(**kwargs)


# ---------------------------------------------------------------------------
def describe_environment() -> str:
    import transformers

    bits = [f"torch {torch.__version__}", f"transformers {transformers.__version__}"]
    for name in ("trl", "peft", "datasets"):
        try:
            bits.append(f"{name} {__import__(name).__version__}")
        except Exception:
            bits.append(f"{name} ?")
    bits.append(f"device {resolve_device()}")
    return " | ".join(bits)
