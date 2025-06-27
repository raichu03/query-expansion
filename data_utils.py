import json
import random
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def load_and_prepare_dataset(json_file_path, test_size=0.20, random_state=42, sample_percentage=0.3):
    """
    Loads the dataset from a JSON file, splits it into train and validation sets,
    and converts it to Hugging Face Dataset objects.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 0.0 < sample_percentage < 1.0:
        print(f"Sampling {sample_percentage * 100}% of the original dataset...")
        num_samples = int(len(data) * sample_percentage)
        random.seed(random_state) # Ensure reproducibility for sampling
        sampled_data = random.sample(data, num_samples)
        print(f"Original size: {len(data)}, Sampled size: {len(sampled_data)}")
        data_to_process = sampled_data
    elif sample_percentage >= 1.0: # Process all data if percentage is 1.0 or greater
        data_to_process = data
    else:
        raise ValueError("sample_percentage must be between 0.0 (exclusive) and 1.0 (inclusive).")

    # Create lists for inputs and outputs
    inputs = [item['input'] for item in data_to_process]
    outputs = [item['output'] for item in data_to_process]

    # Split into train and validation sets
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
        inputs, outputs, test_size=test_size, random_state=random_state
    )

    # Create Hugging Face Dataset objects
    train_dataset = Dataset.from_dict({
        'input': train_inputs,
        'output': train_outputs
    })
    val_dataset = Dataset.from_dict({
        'input': val_inputs,
        'output': val_outputs
    })

    # Combine into a DatasetDict
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

def preprocess_function(raw_data, tokenizer):
    """
    Tokenizes and preprocesses the dataset for the T5 model.
    T5 expects input in a specific format, often with a prefix for the task.
    Here, we combine instruction and input.
    """
    inputs = [f"query expansion input:  {text}" for  text in  raw_data['input']]
    targets = raw_data['output']

    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs