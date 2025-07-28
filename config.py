import torch

# Dataset paths
DATASET_PATH = 'finetune_data.json'

# Model parameters
MAX_SEQ_LENGTH = 1024
DTYPE = None
LOAD_IN_4BIT = True
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

# Training arguments
PER_DEVICE_TRAIN_BATCH_SIZE = 1024
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
NUM_TRAIN_EPOCHS = 5
LEARNING_RATE = 2e-4
FP16 = not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_bf16_supported()
LOGGING_STEPS = 10
OUTPUT_DIR = "output_dir"
OPTIM = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
SEED = 42
EVAL_STRATEGY = "steps"
EVAL_STEPS = 10
SAVE_STEPS = 10
SAVE_TOTAL_LIMIT = 2
PUSH_TO_HUB = False
REPORT_TO = "none"

# LoRA adapter parameters
R = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",]
LORA_ALPHA = 16
LORA_DROPOUT = 0
BIAS = "none"
USE_GRADIENT_CHECKPOINTING = "unsloth"
RANDOM_STATE = 3407
USE_RSLORA = False
LOFTQ_CONFIG = None

# Output directories for saving models
OUTPUT_DIR_ADAPTERS = "./llama3_2_finetuned_adapters"
OUTPUT_DIR_MERGED_MODEL = "./llama3_2_merged_model"

# Prompt templates
ALPACA_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert query expansion assistant. Your goal is to take a user's general search query and expand it with relevant synonyms, related concepts, and clarifications to improve search intent. Be concise but expressive.<|eot_id|><|start_header_id|>user<|end_header_id|>

Input: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

INFERENCE_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert query expansion assistant. Your goal is to take a user's general search query and expand it with relevant synonyms, related concepts, and clarifications to improve search intent. Be concise but expressive.<|eot_id|><|start_header_id|>user<|end_header_id|>

Input: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Inference parameters
MAX_NEW_TOKENS = 256
TOP_P = 0.9
TEMPERATURE = 0.7