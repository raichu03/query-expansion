# Llama-3.2 Query Expansion Fine-tuning
This repository contains a modularized Python codebase for fine-tuning the `unsloth/Llama-3.2-3B-Instruct` model for the task of query expansion. The goal is to take a user's general search query and expand it with relevant synonyms, related concepts, and clarifications to improve search intent.

The original Jupyter Notebook has been refactored into a more maintainable and reusable modular structure, with clear separation of concerns for data handling, model setup, training, and inference.

## Features
* **Modular Design**: Code organized into logical modules for better readability and maintainability.
* **Unsloth Integration**: Utilizes `unsloth` for efficient and fast fine-tuning of Llama models.
* **LoRA Fine-tuning**: Employs Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
* **Custom Prompt Template**: Uses a specific Alpaca-style prompt for query expansion.
* **Train/Eval/Test Split**: Proper dataset splitting for robust evaluation.
* **Inference Script**: Provides a script to test the fine-tuned model with new queries.
* **Model Saving**: Saves both LoRA adapters and a merged 4-bit model.

## Prooject Structure
```
.
├── config.py
├── data_loader.py
├── inference.py
├── main.py
├── model_setup.py
├── training.py
└── finetune_data.json
```
* **`config.py`**: Stores all configuration parameters, including dataset paths, model hyperparameters, training arguments, and prompt templates.
* **`data_loader.py`**: Handles loading data from `finetune_data.json`, formatting it according to the specified prompt templates, and splitting it into training, evaluation, and testing datasets.
* **`model_setup.py`**: Manages the loading of the base Llama-3.2 model and tokenizer, and applies LoRA adapters for fine-tuning.
* **`training.py`**: Contains the logic for setting up and executing the model training process using `trl.SFTTrainer`. It also handles saving the fine-tuned model and its adapters.
* **`inference.py`**: Provides a function to generate responses from the fine-tuned model based on input queries. It includes a `TextStreamer` for real-time output.
* **`main.py`**: The main entry point of the application. It orchestrates the entire workflow: installing dependencies, loading data, setting up the model, initiating training, and running inference tests.
* **`finetune_data.json`**: (Expected) Your dataset in JSON format, containing "instruction", "input", and "output" fields for query expansion.

## Setup and Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/raichu03/query-expansion.git
    cd query-expansion
    ```

2.  **Prepare your environment**:
    The `main.py` script includes commands to install necessary packages. Ensure you have a compatible CUDA environment if you plan to use GPU acceleration.

    ```bash
    pip install -r requirements.txt
    ```
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```
3.  **Ensure CUDA visibility**:
    The `main.py` script sets `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`. Adjust this if you have multiple GPUs and want to use a different one.

## Dataset Preparation
The fine-tuning process expects a JSON file named `finetune_data.json` (or whatever `DATASET_PATH` is set to in `config.py`). This file should contain a list of dictionaries, where each dictionary represents an example for query expansion.

Each example should have the following structure:

```json
[
  {
    "instruction": "Expand the given query.",
    "input": "distance between golden temple and railway station in amritsar",
    "output": "distance between Golden Temple (Amritsar Sri Harimandir Sahib) and nearest railway station (Amritsar Junction), including walking paths and public transportation options."
  },
  {
    "instruction": "Expand the given query.",
    "input": "how often you have a tetanus shot",
    "output": "how often I get vaccinated against tetanus, tetanus shot schedule, tetanus immunization frequency, booster shots for tetanus, tetanus vaccination requirements, tetanus shot interval, adult tetanus vaccination guidelines"
  }
]
```

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Prepare your environment**:
    The `main.py` script includes commands to install necessary packages. Ensure you have a compatible CUDA environment if you plan to use GPU acceleration.

    ```bash
    # These commands are run by main.py, but you can run them manually if preferred
    pip install pip3-autoremove
    pip install torch torchvision torchaudio xformers --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
    pip install unsloth
    pip install --upgrade transformers==4.53.2
    ```
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3.  **Ensure CUDA visibility**:
    The `main.py` script sets `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`. Adjust this if you have multiple GPUs and want to use a different one.

## Dataset Preparation

The fine-tuning process expects a JSON file named `finetune_data.json` (or whatever `DATASET_PATH` is set to in `config.py`). This file should contain a list of dictionaries, where each dictionary represents an example for query expansion.

Each example should have the following structure:

```json
[
  {
    "instruction": "Expand the given query.",
    "input": "distance between golden temple and railway station in amritsar",
    "output": "distance between Golden Temple (Amritsar Sri Harimandir Sahib) and nearest railway station (Amritsar Junction), including walking paths and public transportation options."
  },
  {
    "instruction": "Expand the given query.",
    "input": "how often you have a tetanus shot",
    "output": "how often I get vaccinated against tetanus, tetanus shot schedule, tetanus immunization frequency, booster shots for tetanus, tetanus vaccination requirements, tetanus shot interval, adult tetanus vaccination guidelines"
  }
]
```
The `data_loader.py` module is responsible for loading this data and formatting it with the `ALPACA_PROMPT` for training and `INFERENCE_PROMPT_TEMPLATE` for inference, as defined in `config.py`.

## Configuration
All configurable parameters are centralized in `config.py`. You can adjust these values to suit your specific needs:

`DATASET_PATH`: Path to your fine-tuning dataset.

Model Parameters: `MAX_SEQ_LENGTH`, `DTYPE`, `LOAD_IN_4BIT`, `MODEL_NAME`.

Training Arguments: `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, etc.

LoRA Parameters: `R`, `TARGET_MODULES`, `LORA_ALPHA`, `LORA_DROPOUT`, etc.

Output Directories: `OUTPUT_DIR_ADAPTERS`, `OUTPUT_DIR_MERGED_MODEL`.

Prompt Templates: `ALPACA_PROMPT` (for training), `INFERENCE_PROMPT_TEMPLATE` (for inference).

Inference Parameters: `MAX_NEW_TOKENS`, `TOP_P`, `TEMPERATURE`.

## Usage
To start the fine-tuning process, simply run the `main.py` script.

After training, `main.py` will automatically proceed to run inference on a small subset of the test data (default 5 samples). The generated responses will be printed to the console and saved to `output_data.json`.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details