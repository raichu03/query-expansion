import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

def initialize_model_and_tokenizer(model_name):
    """
    Initializes and returns the pre-trained T5 tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def configure_peft_model(model):
    """
    Configures the model with PEFT (LoRA) for fine-tuning.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model