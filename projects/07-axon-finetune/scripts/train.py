#!/usr/bin/env python3
"""
LoRA fine-tune SmolLM2-360M-Instruct for Axon portfolio assistant.
Trains on RTX 2070 SUPER (8GB, SM_75 — fp16 only, no bf16).
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# Paths
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATASET_PATH = os.environ.get("DATASET_PATH", "/opt/data/dataset.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/lora-adapter")
HF_CACHE = os.environ.get("HF_HOME", "/workspace/hf_cache")

# Training hyperparameters
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # effective batch = 16
MAX_SEQ_LENGTH = 512
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
EVAL_SPLIT = 0.1


def main():
    print("=" * 60)
    print("Axon LoRA Fine-Tune — SmolLM2-360M-Instruct")
    print("=" * 60)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected, training on CPU (will be slow)")

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in bf16 (5090 SM_120 supports bf16)
    print(f"Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Model dtype: {model.dtype}")

    # LoRA config — target all linear layers
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Total examples: {len(dataset)}")

    # Split train/eval
    split = dataset.train_test_split(test_size=EVAL_SPLIT, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_SEQ_LENGTH,
        fp16=False,
        bf16=True,  # 5090 SM_120 supports bf16
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # No wandb for this small project
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print("=" * 60 + "\n")

    train_result = trainer.train()

    # Save best model
    print("\nSaving best LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Print metrics
    metrics = train_result.metrics
    print(f"\nTraining complete!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Train runtime: {metrics.get('train_runtime', 0):.1f}s")
    print(f"  Samples/sec: {metrics.get('train_samples_per_second', 0):.1f}")

    # Eval
    eval_metrics = trainer.evaluate()
    print(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")

    # Save metrics
    all_metrics = {**metrics, **eval_metrics}
    with open(os.path.join(OUTPUT_DIR, "train_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nLoRA adapter saved to {OUTPUT_DIR}")
    print(f"GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
