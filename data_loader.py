from datasets import load_dataset, Dataset
import pandas as pd
import json
from config import ALPACA_PROMPT, INFERENCE_PROMPT_TEMPLATE

def load_and_process_dataset(dataset_path, is_training=True):
    """
    Loads the dataset and applies the specified formatting function.

    Args:
        dataset_path (str): Path to the JSON dataset file.
        is_training (bool): If True, uses ALPACA_PROMPT for training.
                            If False, uses INFERENCE_PROMPT_TEMPLATE for testing/inference.

    Returns:
        Dataset: The processed dataset.
    """
    try:
        raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    except Exception as e:
        print(f"Error loading dataset directly: {e}. Trying pandas fallback.")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        raw_dataset = Dataset.from_pandas(pd.DataFrame(data))

    print(f"Original dataset size: {len(raw_dataset)} rows.")

    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for input_text, output in zip(inputs, outputs):
            if is_training:
                text = ALPACA_PROMPT.format(input=input_text, output=output)
            else:
                text = INFERENCE_PROMPT_TEMPLATE.format(input=input_text)
            texts.append(text)
        return {"text": texts}

    processed_dataset = raw_dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=4,
        remove_columns=raw_dataset.column_names
    )
    return processed_dataset

def split_dataset(processed_dataset, train_ratio=0.2, eval_ratio=0.5, test_ratio=0.5):
    """
    Splits the processed dataset into training, evaluation, and test sets.

    Args:
        processed_dataset (Dataset): The dataset already formatted with prompts.
        train_ratio (float): Proportion of the dataset to use for the initial train split.
        eval_ratio (float): Proportion of the remaining data to use for the evaluation set.
        test_ratio (float): Proportion of the remaining data to use for the test set.

    Returns:
        tuple: (train_dataset, eval_dataset, test_dataset)
    """
    # First split to get initial train and a combined eval/test pool
    train_test_split = processed_dataset.train_test_split(test_size=(1 - train_ratio))
    train_dataset = train_test_split["train"]
    remaining_dataset = train_test_split["test"]

    # Split the remaining dataset into evaluation and test
    eval_test_split = remaining_dataset.train_test_split(test_size=test_ratio) # test_size here is relative to remaining_dataset
    eval_dataset = eval_test_split["train"]
    test_dataset = eval_test_split["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, eval_dataset, test_dataset