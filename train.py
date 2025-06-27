import os
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from data_utils import load_and_prepare_dataset, preprocess_function
from model_utils import initialize_model_and_tokenizer, configure_peft_model

def main():
    dataset_path = "/kaggle/input/finetune-llm/finetune_data.json"
    model_name = "t5-base"
    output_dir = "./t5_query_expansion_lora"
    sample_percentage = 0.3

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    raw_datasets = load_and_prepare_dataset(dataset_path, sample_percentage=sample_percentage)
    print("Dataset loaded successfully:")
    print(raw_datasets)
    print(raw_datasets['train'][0])

    tokenizer, model = initialize_model_and_tokenizer(model_name)

    print("\nTokenizing and preprocessing dataset...")
    tokenized_datasets = raw_datasets.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    print("Preprocessing complete.")
    print(tokenized_datasets['train'][0])

    tokenized_datasets = tokenized_datasets.remove_columns([ "input", "output"])

    model = configure_peft_model(model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=16,
        learning_rate=1e-6,
        num_train_epochs=10,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
        gradient_accumulation_steps=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        label_names=["labels"],
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()
    print("Training complete.")

    final_model_save_path = f"{output_dir}/final_lora_adapters"
    model.save_pretrained(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    print(f"LoRA adapters and tokenizer saved to: {final_model_save_path}")

if __name__ == "__main__":
    main()