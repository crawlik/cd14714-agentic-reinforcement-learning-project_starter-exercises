# SFT Fine-Tuning - Starter Code
# Module 2: Practical Fine-Tuning with PEFT

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import pandas as pd
import json

class SFTConfig:
    """Configuration for SFT fine-tuning."""
    base_model_name = "google/gemma-3-270m-it"
    output_model_path = "models/sft_prediction_model_gemma_270m"
    sft_data_path = "../../../lesson-1-supervised-fine-tuning/exercises/starter/clinical_sft_dataset.csv"
    
    # LoRA parameters - TODO: Configure these appropriately
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.15
    lora_target_modules = ["q_proj", "v_proj"]
    
    # Training parameters - TODO: Tune these
    num_train_epochs = 20
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 3e-5

def load_and_format_dataset(csv_path: str):
    """
    Load and format the SFT dataset for training.
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    examples = []
    for _, row in df.iterrows():
        prompt = f"Determine eligibility: {row['patient_summary']}"
        completion = f"{row['eligibility_status']}"
        examples.append({"text": f"{prompt}\n{completion}"})

    return examples

def configure_lora():
    """
    Configure LoRA parameters for efficient fine-tuning.
    """
    return LoraConfig(
        r=SFTConfig.lora_r,
        lora_alpha=SFTConfig.lora_alpha,
        lora_dropout=SFTConfig.lora_dropout,
        target_modules=SFTConfig.lora_target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

def run_sft_training(config: SFTConfig, train_examples):
    """
    Run the SFT training process.
    """
    print("Starting SFT training...")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name)

    train_dataset = Dataset.from_list(train_examples)
    peft_config = configure_lora()

    training_args = TrainingArguments(
        output_dir=config.output_model_path,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(config.output_model_path)

    return trainer.model, tokenizer

def validate_model(model, tokenizer, val_examples):
    """
    Validate the fine-tuned model on validation examples.
    """
    print("Validating model...")

    correct = 0
    confidences = []

    model.eval()
    for example in val_examples:
        prompt, actual_status = example["text"].rsplit("\n", 1)
        actual_status = actual_status.strip()

        input_ids = tokenizer.encode(prompt + "\n", return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if outputs.scores:
            first_token_probs = torch.softmax(outputs.scores[0][0], dim=-1)
            confidences.append(first_token_probs.max().item())

        if actual_status in prediction:
            correct += 1

    accuracy = correct / len(val_examples) if val_examples else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return accuracy, avg_confidence

if __name__ == "__main__":
    config = SFTConfig()

    # Load and format dataset
    examples = load_and_format_dataset(config.sft_data_path)

    if not examples:
        print("No training examples found. Please generate the dataset first.")
        exit(1)

    # Hold out a few examples for validation
    val_size = max(1, len(examples) // 5)
    train_examples = examples[:-val_size]
    val_examples = examples[-val_size:]

    # Configure LoRA
    peft_config = configure_lora()

    # Run training
    model, tokenizer = run_sft_training(config, train_examples)

    print("SFT training completed!")

    # Validate on held-out examples
    accuracy, confidence = validate_model(model, tokenizer, val_examples)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Average confidence: {confidence:.2%}")
