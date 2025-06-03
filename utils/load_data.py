# In load_data.py

import pandas as pd

def extract_string_batches(tsv_filepath, string_column_name, pandas_chunk_size, yield_batch_size, skip_n_values=0):
    """
    Extracts string values from a specified column in a TSV file in batches,
    allowing skipping of initial valid values, and excluding 'nan' strings.

    Args:
        tsv_filepath (str): The path to the TSV file.
        string_column_name (str): The name of the column containing string values.
        pandas_chunk_size (int): The number of rows to read from TSV at a time using pandas.
        yield_batch_size (int): The number of valid string values to yield in each batch.
        skip_n_values (int): The number of valid (non-'nan') string values to skip 
                             from the beginning before starting to yield.

    Yields:
        list: A list of string values, representing a single batch, without 'nan' values.
    """
    values_to_skip_remaining = skip_n_values
    output_buffer = []
    
    try:
        for chunk in pd.read_csv(tsv_filepath, sep='\t', chunksize=pandas_chunk_size, low_memory=False):
            if string_column_name not in chunk.columns:
                print(f"Warning: Column '{string_column_name}' not found in a chunk. Skipping chunk.")
                continue

            string_values_from_chunk = chunk[string_column_name].astype(str)
            
            for s_val in string_values_from_chunk:
                if s_val.lower() == 'nan': # Case-insensitive check for 'nan' string
                    continue

                # This is a valid string value
                if values_to_skip_remaining > 0:
                    values_to_skip_remaining -= 1
                    continue
                
                # If not skipping, add to buffer
                output_buffer.append(s_val)
                if len(output_buffer) == yield_batch_size:
                    yield output_buffer
                    output_buffer = []
        
        if output_buffer: # Yield any remaining values in the buffer
            yield output_buffer
            
    except FileNotFoundError:
        print(f"Error: TSV file not found at {tsv_filepath}")
        return # Stop iteration
    except pd.errors.EmptyDataError:
        print(f"Error: TSV file at {tsv_filepath} is empty.")
        return # Stop iteration
    except Exception as e:
        print(f"An unexpected error occurred while reading {tsv_filepath}: {e}")
        return # Stop iteration