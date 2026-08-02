# Structured Fine Tuning -- Stage 1: SFT probability predictor
#
# Trains google/gemma-3-270m-it with LoRA to emit
#     {"probability": 0.XX, "reason": "word"}
# given a two-attendee description and a conference time slot.
#
#   python starter_sft.py --smoke     # 2 min pipeline check, throwaway model
#   python starter_sft.py             # full run, saves the graded adapter
#
from dataclasses import dataclass, field, asdict
from datasets import Dataset
from datetime import datetime
import argparse
import json
import numpy as np
import os
import pandas as pd
from peft import LoraConfig
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple

#### Core data classes
from data_classes import (
    safe_extract_json,
    MeetingPredictor,
    build_prediction_prompt,
    wrap_gemma_turn,
)

#### Version bridge: the classroom image and the local Mac run different TRL majors
from compat import (
    describe_environment,
    make_sft_trainer,
    make_sft_training_args,
    resolve_device,
    resolve_precision,
)


# ===========================================================================
# Hyperparameters
# ===========================================================================
@dataclass
class SFTConfig:
    """Stage 1 hyperparameters.

    Choices and why:

    lora_r=16 / lora_alpha=32
        Gemma-270m is small, and the task is essentially scalar regression
        rendered as text. r=8 underfits the long tail of the probability
        distribution; r=16 with the conventional alpha=2r converged noticeably
        faster in testing without the instability r=32 introduced. Only 2 of the
        ~270M parameters' projections are adapted (q_proj/v_proj), so this is
        still well under 1% trainable.

    lora_dropout=0.1
        2000 examples over 12 epochs is enough repetition to memorise. 0.1 is
        the standard starting point; raise toward 0.15 if train loss falls far
        below validation quality.

    num_train_epochs=12
        With batch 2 x grad-accum 4, one epoch over 1600 training rows is 200
        optimizer steps. Fewer than ~8 epochs and the model has not learned to
        close the JSON reliably; past ~15 it starts memorising individual
        descriptions and validation correlation drops.

    learning_rate=3e-5
        Mid-range for LoRA fine-tuning. 5e-5 produced loss spikes on this
        dataset; 1e-5 needed roughly twice the epochs for the same loss.

    weight_decay=0.01
        Light regularisation. The target vocabulary is tiny (a probability and
        one word) so aggressive decay mostly just slows convergence.
    """

    base_model_name: str = "google/gemma-3-270m-it"
    output_model_path: str = "models/sft_prediction_model_gemma_270m"
    sft_data_path: str = "data/sft_training_data.csv"

    # ---- LoRA ----
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # ---- precision ----
    use_4bit: bool = False
    fp16: bool = False
    bf16: bool = False

    # ---- optimisation ----
    num_train_epochs: int = 12
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    logging_steps: int = 10
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine_with_restarts"
    warmup_ratio: float = 0.03
    seed: int = 42

    # ---- sequence / validation ----
    # p99 of input_text is ~310 tokens and the longest row is ~400, so 512
    # truncates nothing while keeping attention cost down.
    max_seq_length: int = 512

    # Early stopping guards against OVERFITTING, not slow convergence: if the
    # epoch's mean training loss falls below this the model has started
    # memorising the reason words rather than learning the probability mapping.
    # Note the reason field is a ~280-way long tail of near-synonyms, so a
    # genuinely well-generalising model plateaus around 0.6-0.9 here. Dropping
    # under 0.4 means memorisation.
    early_stopping_loss: float = 0.4

    # {"probability": 0.79, "reason": "engagement"} is ~20 tokens. 64 leaves
    # slack for a longer reason word without letting the model ramble past the
    # closing brace and waste generation time during validation.
    val_max_new_tokens: int = 64

    val_samples_per_epoch: int = 25


# ===========================================================================
# Data
# ===========================================================================
def normalise_reason(reason: str) -> str:
    """The CSV contains 'engagement' and 'engagement.' as distinct labels
    (129 and 49 rows). Collapsing the punctuation variants removes a chunk of
    pointless label noise from the target distribution."""
    return str(reason).strip().rstrip(".").strip().lower()


def load_sft_examples(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    examples = []
    for _, row in df.iterrows():
        examples.append(
            {
                "input_text": row["input_text"],
                "target_json_output": {
                    "probability": round(float(row["target_probability"]), 2),
                    "reason": normalise_reason(row["target_reason"]),
                },
                "ground_truth_prob": float(row["ground_truth_prob"]),
            }
        )
    return examples


def format_for_training(example: Dict) -> Dict[str, str]:
    """Gemma chat format. The user turn is verbatim the CSV's input_text so that
    Stage 2's MeetingPredictor sees exactly the format trained on."""
    target_str = json.dumps(example["target_json_output"])
    text = (
        f"<start_of_turn>user\n{example['input_text']}<end_of_turn>\n"
        f"<start_of_turn>model\n{target_str}<end_of_turn>"
    )
    return {"text": text}


# ===========================================================================
# Validation
# ===========================================================================
def predict_raw(model, tokenizer, input_text: str, device: str, max_new_tokens: int) -> str:
    """Greedy-decode one completion for a raw CSV input_text and return only the
    newly generated tokens (not the echoed prompt)."""
    prompt = wrap_gemma_turn(input_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def score_predictions(predictions: List[float], ground_truths: List[float]) -> Dict[str, float]:
    """Correlation, MAE and the directional YES/NO accuracy the rubric grades.

    Accuracy is the fraction of examples where the model lands on the same side
    of the 0.5 decision boundary as the simulator -- i.e. would the agent make
    the same YES/NO call. A coin flip on this dataset scores ~50% because the
    label balance is 47.6% YES.
    """
    predictions = np.asarray(predictions, dtype=float)
    ground_truths = np.asarray(ground_truths, dtype=float)

    if len(predictions) < 2:
        return {"correlation": 0.0, "mae": float("nan"), "accuracy": 0.0, "n": len(predictions)}

    if predictions.std() == 0 or ground_truths.std() == 0:
        correlation = 0.0  # a constant predictor has undefined correlation
    else:
        correlation = float(np.corrcoef(predictions, ground_truths)[0, 1])

    mae = float(np.mean(np.abs(predictions - ground_truths)))
    accuracy = float(np.mean((predictions > 0.5) == (ground_truths > 0.5)) * 100)
    return {"correlation": correlation, "mae": mae, "accuracy": accuracy, "n": len(predictions)}


def validate_model(predictor: MeetingPredictor, val_examples: List[Dict], num_samples: int = 20):
    """Validate a *saved* adapter through the same MeetingPredictor class that
    Stage 2 uses as a tool, so a pass here proves the whole serving path works
    end to end -- not just that the in-memory trainer object can generate.
    """
    predictions: List[float] = []
    ground_truths: List[float] = []
    parse_failures = 0

    for i, example in enumerate(val_examples[:num_samples]):
        try:
            response_text = predict_raw(
                predictor.model,
                predictor.tokenizer,
                example["input_text"],
                predictor.device,
                max_new_tokens=64,
            )
            pred_result = safe_extract_json(response_text)
            predicted_prob = float(np.clip(float(pred_result["probability"]), 0.0, 1.0))

            predictions.append(predicted_prob)
            ground_truths.append(float(example["ground_truth_prob"]))

            if i % 10 == 0:
                print(
                    f"  sample {i:>3}: pred={predicted_prob:.3f} "
                    f"true={example['ground_truth_prob']:.3f} "
                    f"reason={pred_result.get('reason', '?')}"
                )
        except Exception as e:
            parse_failures += 1
            if parse_failures <= 3:
                print(f"  sample {i:>3}: FAILED ({type(e).__name__}: {str(e)[:80]})")
            continue

    if parse_failures:
        print(f"  {parse_failures}/{min(num_samples, len(val_examples))} responses unparseable")

    metrics = score_predictions(predictions, ground_truths)
    return (
        metrics["correlation"],
        metrics["mae"],
        np.asarray(predictions),
        np.asarray(ground_truths),
    )


# ===========================================================================
# Training
# ===========================================================================
def run_sft_fine_tuning(
    config: SFTConfig,
    train_examples: Optional[List[Dict]] = None,
    val_examples: Optional[List[Dict]] = None,
) -> Dict:
    if train_examples is None:
        train_examples = load_sft_examples(config.sft_data_path)
    formatted_examples = [format_for_training(e) for e in train_examples]

    dataset = Dataset.from_list(formatted_examples)
    print(f"Dataset created with {len(dataset)} examples")

    print(f"\n Configuring Model ('{config.base_model_name}') for SFT ")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name, trust_remote_code=True, attn_implementation="eager"
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Causal LM, not sequence classification: we are not picking one of k labels,
    # we are asking the model to *write* a JSON object token by token, where each
    # token is conditioned on everything before it. The probability digits are
    # generated the same way the reason word is. That also means the SFT model
    # inherits the base model's instruction-following, which is what lets Stage 2
    # call it as a drop-in tool rather than a bespoke regression head.
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    device = resolve_device()
    precision = resolve_precision(device)
    config.bf16, config.fp16 = precision["bf16"], precision["fp16"]

    # One epoch per trainer.train() call so the per-epoch validation and early
    # stopping below are meaningful. (The original starter set
    # num_train_epochs=N *and* wrapped train() in a range(N) loop, which trains
    # N^2 epochs.)
    training_args = make_sft_training_args(
        output_dir=config.output_model_path,
        num_train_epochs=1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        group_by_length=True,
        save_strategy="no",          # we save the adapter ourselves, once
        report_to="none",            # no wandb prompt in a headless workspace
        seed=config.seed,
        dataset_text_field="text",
        max_length=config.max_seq_length,
        packing=False,
    )

    trainer = make_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
    )

    print("\n Starting Supervised Fine-Tuning ")
    print(f" {describe_environment()}")
    history = {"train_loss": [], "correlation": [], "mae": [], "accuracy": []}
    best_correlation = -np.inf

    for epoch in range(config.num_train_epochs):
        print(f"\n=== EPOCH {epoch + 1}/{config.num_train_epochs} ===")

        # Rebuild the LR schedule each epoch (the optimizer and its Adam moments
        # persist). With cosine_with_restarts this is exactly one cosine cycle
        # per epoch; without the reset the schedule is exhausted after epoch 1
        # and the LR sits at ~0 for the rest of training.
        trainer.lr_scheduler = None
        trainer.train()

        log = [h for h in trainer.state.log_history if "train_loss" in h]
        current_loss = log[-1]["train_loss"] if log else float("nan")
        history["train_loss"].append(current_loss)
        print(f"Training Loss: {current_loss:.4f}")

        if current_loss < config.early_stopping_loss:
            print(
                f"Loss {current_loss:.4f} dropped below {config.early_stopping_loss}. "
                "Potential overfitting detected. Stopping training."
            )
            break

        if val_examples:
            print("Running validation...")
            trainer.model.eval()
            val_predictions, val_ground_truths = [], []

            for i, val_example in enumerate(val_examples[: config.val_samples_per_epoch]):
                response_text = predict_raw(
                    trainer.model,
                    tokenizer,
                    val_example["input_text"],
                    device,
                    max_new_tokens=config.val_max_new_tokens,
                )
                try:
                    parsed_json = safe_extract_json(response_text)
                    predicted_prob = float(np.clip(float(parsed_json["probability"]), 0.0, 1.0))
                    val_predictions.append(predicted_prob)
                    val_ground_truths.append(float(val_example["ground_truth_prob"]))
                    if i % 10 == 0:
                        print(
                            f"  Sample {i}: Pred={predicted_prob:.3f}, "
                            f"True={val_example['ground_truth_prob']:.3f}"
                        )
                except Exception as e:
                    print(f"  VALIDATION FAILED on sample {i}: {str(e)[:100]}")
                    continue

            trainer.model.train()
            metrics = score_predictions(val_predictions, val_ground_truths)
            history["correlation"].append(metrics["correlation"])
            history["mae"].append(metrics["mae"])
            history["accuracy"].append(metrics["accuracy"])
            print(
                f"Validation - Correlation: {metrics['correlation']:.3f}, "
                f"MAE: {metrics['mae']:.3f}, Accuracy: {metrics['accuracy']:.1f}% "
                f"(parsed {metrics['n']}/{min(config.val_samples_per_epoch, len(val_examples))})"
            )
            if metrics["correlation"] > best_correlation:
                best_correlation = metrics["correlation"]

    os.makedirs(config.output_model_path, exist_ok=True)
    trainer.save_model(config.output_model_path)
    tokenizer.save_pretrained(config.output_model_path)  # Stage 2 loads from here
    print(f"SFT model saved to '{config.output_model_path}'")

    run_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "environment": describe_environment(),
        "hyperparameters": {k: v for k, v in asdict(config).items()},
        "epochs_completed": len(history["train_loss"]),
        "history": history,
        "best_correlation": None if best_correlation == -np.inf else best_correlation,
    }
    with open(os.path.join(config.output_model_path, "training_run.json"), "w") as fh:
        json.dump(run_record, fh, indent=2)
    print("Run record written to training_run.json (paste this into the report)")

    return run_record


# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 1: SFT probability predictor")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to prove the pipeline works; do not submit the result")
    parser.add_argument("--epochs", type=int, default=None, help="override num_train_epochs")
    parser.add_argument("--train-size", type=int, default=None, help="cap training examples")
    parser.add_argument("--output", type=str, default=None, help="override adapter output dir")
    parser.add_argument("--skip-final-check", action="store_true",
                        help="skip reloading the saved adapter for end-to-end validation")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print(" STEP 1: Loading SFT Data for Prediction Model ")
    sft_config = SFTConfig()
    if args.output:
        sft_config.output_model_path = args.output

    csv_path = sft_config.sft_data_path
    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing {csv_path}. Run this script from the project/ directory.")

    sft_dataset_examples = load_sft_examples(csv_path)
    print(f"Loaded {len(sft_dataset_examples)} existing examples")

    print(f"\n Sample Training Examples ")
    for i in range(min(2, len(sft_dataset_examples))):
        example = sft_dataset_examples[i]
        print(f"\nExample {i + 1}:")
        print(f"Input: {example['input_text'][:200]}...")
        print(f"Target: {example['target_json_output']}")
        print(f"Ground Truth: {example['ground_truth_prob']:.3f}")

    if args.smoke:
        print("\n*** SMOKE TEST: 1 epoch on 64 examples, throwaway output ***")
        sft_config.num_train_epochs = 1
        sft_config.val_samples_per_epoch = 4
        sft_config.output_model_path = args.output or "models/_smoke_test_adapter"
        sft_dataset_examples = sft_dataset_examples[:80]
    if args.epochs is not None:
        sft_config.num_train_epochs = args.epochs

    print(f"\n STEP 2: Fine-Tuning the Gemma-270m Prediction Model ")
    print(f" {describe_environment()}")

    split_idx = int(0.8 * len(sft_dataset_examples))
    train_examples = sft_dataset_examples[:split_idx]
    val_examples = sft_dataset_examples[split_idx:]
    if args.train_size:
        train_examples = train_examples[: args.train_size]

    print(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")

    run_sft_fine_tuning(sft_config, train_examples, val_examples)

    if not args.skip_final_check:
        print("\n STEP 3: Reloading the saved adapter and validating end to end ")
        print(" (this is the exact path the Stage 2 prediction tool will use)")
        try:
            predictor = MeetingPredictor(sft_config.output_model_path)
            correlation, mae, preds, truths = validate_model(predictor, val_examples, num_samples=50)
            metrics = score_predictions(list(preds), list(truths))
            print(
                f"\nFINAL  correlation={correlation:.3f}  MAE={mae:.3f}  "
                f"accuracy={metrics['accuracy']:.1f}%  (n={metrics['n']})"
            )
            with open(os.path.join(sft_config.output_model_path, "final_validation.json"), "w") as fh:
                json.dump(metrics, fh, indent=2)
        except Exception as e:
            print(f"End-to-end check failed: {type(e).__name__}: {e}")
            print("The adapter is still saved; investigate before running Stage 2.")

    print("\n STAGE 2 would begin here ")
    print(" With your fine tuned model saved, we can now start building our agent traces.")


if __name__ == "__main__":
    main()
