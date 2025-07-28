from transformers import TextStreamer
from config import INFERENCE_PROMPT_TEMPLATE, MAX_NEW_TOKENS, TOP_P, TEMPERATURE
import torch

def generate_response(model, tokenizer, input_text):
    """
    Generates a response from the model based on the instruction and input.

    Args:
        model: The trained model.
        tokenizer: The tokenizer for the model.
        input_text (str): The formatted input prompt.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated assistant's response.
    """
    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        add_special_tokens=False
    ).to("cuda")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n--- Input Query ---")
    print(f"Input: {input_text}")
    print(f"--- Generated Output ---")

    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        do_sample=True,
        top_p=TOP_P,
        temperature=TEMPERATURE,
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to extract only the assistant's response if needed
    assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if assistant_prefix in decoded_output:
        assistant_response = decoded_output.split(assistant_prefix, 1)[1].split("<|eot_id|>")[0].strip()
    else:
        assistant_response = decoded_output.strip()

    return assistant_response