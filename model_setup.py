from unsloth import FastLanguageModel
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT,
    R, TARGET_MODULES, LORA_ALPHA, LORA_DROPOUT, BIAS,
    USE_GRADIENT_CHECKPOINTING, RANDOM_STATE, USE_RSLORA, LOFTQ_CONFIG
)

def setup_model_and_tokenizer():
    """
    Loads the FastLanguageModel and tokenizer, and applies LoRA adapters.

    Returns:
        tuple: (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = R,
        target_modules = TARGET_MODULES,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        bias = BIAS,
        use_gradient_checkpointing = USE_GRADIENT_CHECKPOINTING,
        random_state = RANDOM_STATE,
        use_rslora = USE_RSLORA,
        loftq_config = LOFTQ_CONFIG,
    )
    print("Model and Tokenizer set up with LoRA adapters.")
    return model, tokenizer