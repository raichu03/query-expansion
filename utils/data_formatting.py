import pandas as pd
import json

def prepare_query_expansion_data(csv_file_path, original_query_col, expanded_query_col):
    """
    Reads a CSV, formats it for query expansion finetuning, and returns a list of dictionaries.

    Args:
        csv_file_path (str): Path to your CSV file.
        original_query_col (str): Name of the column containing original queries.
        expanded_query_col (str): Name of the column containing expanded queries.

    Returns:
        list: A list of dictionaries, each formatted for model finetuning.
    """
    
    df = pd.read_csv(csv_file_path)
    
    formatted_data = []
    for index, row in df.iterrows():
        original_query = row[original_query_col]
        expanded_query = row[expanded_query_col]
        
        # Define the instruction. Be consistent with this!
        # This tells the model what to do.
        instruction = "Expand the user's query with synonyms, related concepts, and clarifications to better match search intent. Keep it concise but expressive."
        
        formatted_data.append({
            "instruction": instruction,
            "input": original_query,
            "output": expanded_query
        })
        
    return formatted_data

if __name__=="__main__":
    
    csv_file = 'data/all_queries.csv' 
    original_col = 'query'
    expanded_col = 'expanded'
    
    prepared_dataset = prepare_query_expansion_data(csv_file, original_col, expanded_col)
    
    with open('data/finetune_data.json', 'w', encoding='utf-8') as f:
        json.dump(prepared_dataset, f, ensure_ascii=False, indent=4)

    print(f"\nDataset prepared and saved to 'finetune_data.json' with {len(prepared_dataset)} examples.")