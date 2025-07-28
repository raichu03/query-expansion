from transformers import TrainingArguments
from trl import SFTTrainer
from config import (
    PER_DEVICE_TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, WARMUP_STEPS,
    NUM_TRAIN_EPOCHS, LEARNING_RATE, FP16, BF16, LOGGING_STEPS, OUTPUT_DIR,
    OPTIM, WEIGHT_DECAY, LR_SCHEDULER_TYPE, SEED, EVAL_STRATEGY, EVAL_STEPS,
    SAVE_STEPS, SAVE_TOTAL_LIMIT, PUSH_TO_HUB, REPORT_TO, MAX_SEQ_LENGTH,
    OUTPUT_DIR_ADAPTERS, OUTPUT_DIR_MERGED_MODEL
)

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """
    Sets up and runs the training of the model.

    Args:
        model: The model to be trained.
        tokenizer: The tokenizer for the model.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
    """
    training_arguments = TrainingArguments(
        per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        warmup_steps = WARMUP_STEPS,
        num_train_epochs = NUM_TRAIN_EPOCHS,
        learning_rate = LEARNING_RATE,
        fp16 = FP16,
        bf16 = BF16,
        logging_steps = LOGGING_STEPS,
        output_dir = OUTPUT_DIR,
        optim = OPTIM,
        weight_decay = WEIGHT_DECAY,
        lr_scheduler_type = LR_SCHEDULER_TYPE,
        seed = SEED,
        eval_strategy = EVAL_STRATEGY,
        eval_steps = EVAL_STEPS,
        save_steps = SAVE_STEPS,
        save_total_limit = SAVE_TOTAL_LIMIT,
        push_to_hub = PUSH_TO_HUB,
        report_to = REPORT_TO,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 4,
        packing = False,
        args = training_arguments,
    )

    print("\nStarting training...")
    trainer.train()
    print("Training complete!")

    # Save LoRA adapters
    model.save_pretrained_merged(OUTPUT_DIR_ADAPTERS, tokenizer, save_method = "adapters")
    print(f"LoRA adapters saved to {OUTPUT_DIR_ADAPTERS}")

    # Save merged 4bit model
    model.save_pretrained_merged(OUTPUT_DIR_MERGED_MODEL, tokenizer, save_method = "merged_4bit_forced")
    print(f"Merged model saved to {OUTPUT_DIR_MERGED_MODEL}")