import os
import torch
import json
from data_loader import load_and_process_dataset, split_dataset
from model_setup import setup_model_and_tokenizer
from training import train_model
from inference import generate_response
from config import DATASET_PATH, OUTPUT_DIR_ADAPTERS, OUTPUT_DIR_MERGED_MODEL, INFERENCE_PROMPT_TEMPLATE

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Installing required packages...")
    os.system("pip install pip3-autoremove")
    os.system("pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124")
    os.system("pip install unsloth")
    os.system("pip install --upgrade transformers==4.53.2")
    print("Packages installed.")

    print("\n--- Preparing Training/Evaluation Data ---")
    processed_train_eval_dataset = load_and_process_dataset(DATASET_PATH, is_training=True)
    train_dataset, eval_dataset, _ = split_dataset(processed_train_eval_dataset, train_ratio=0.2, eval_ratio=0.5, test_ratio=0) # No test split here

    print("\n--- Setting up Model ---")
    model, tokenizer = setup_model_and_tokenizer()

    print("\n--- Starting Model Training ---")
    train_model(model, tokenizer, train_dataset, eval_dataset)

    print("\n--- Preparing Test Data for Inference ---")
    raw_test_dataset_full = load_and_process_dataset(DATASET_PATH, is_training=False)
    _, _, test_dataset = split_dataset(raw_test_dataset_full, train_ratio=0.2, eval_ratio=0.5, test_ratio=0.5)

    print("\n--- Running Inference Tests ---")
    generated_data = []
    num_test_samples = min(5, len(test_dataset)) # Limit to 5 samples or less if dataset is smaller

    for i in range(num_test_samples):
        input_text_for_inference = test_dataset[i]['text']
        generated_text = generate_response(model, tokenizer, input_text_for_inference)
        new_data = {
            'input': input_text_for_inference,
            'output': generated_text
        }
        generated_data.append(new_data)
        print("-" * 50)

    output_file_path = "output_data.json"
    with open(output_file_path, 'w') as json_file:
        json.dump(generated_data, json_file, indent=4)
    print(f"\nInference results saved to {output_file_path}")

if __name__ == "__main__":
    main()