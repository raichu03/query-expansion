import ollama
import asyncio
import csv
import aiofiles # For asynchronous file I/O
from tqdm.asyncio import tqdm # For a nice progress bar for async operations
import os # For checking file existence

# Assuming extract_string_batches_resumable is in load_data.py
from utils.load_data import extract_string_batches

async def count_csv_data_rows(filepath, header_line_content=None):
    """
    Counts the number of data rows in a CSV file, excluding a specific header.
    A simple line count is used for speed. More robust parsing can be added if needed.
    """
    count = 0
    try:
        async with aiofiles.open(filepath, mode='r', newline='', encoding='utf-8') as f:
            first_line = await f.readline()
            if not first_line: # Empty file
                return 0
            
            # Check if the first line is the expected header
            if header_line_content and first_line.strip() == header_line_content.strip():
                # It's the header, proceed to count remaining lines
                pass
            elif first_line.strip(): # It's a data line (no header or different header)
                count = 1
            
            # Count remaining lines
            async for _ in f:
                count += 1
        return count
    except FileNotFoundError:
        return 0

async def expand_query(query, semaphore):
    """
    Expands a single user query using the Ollama Llama model.
    Includes error handling and uses a semaphore for concurrency control.
    """
    client = ollama.AsyncClient()
    # Ensure query is a string, handle potential non-string types if necessary
    if not isinstance(query, str):
        # print(f"Warning: Received non-string query: {query} (type: {type(query)}). Skipping.")
        return str(query) if query is not None else "Invalid Query", "Error: Non-string query received"


    prompt = f'''
        Expand the given user query to improve search engine results. 
        Include synonyms, related concepts, and clarify ambiguous 
        terms. The goal is to better reflect the user's actual 
        intent and information need. Keep the expanded query concise but expressive.
        Original query: {query}. Respond with only the expanded query, nothing else.
        '''
    
    async with semaphore:  # Acquire semaphore before making the API call
        try:
            response = await client.chat(
                model="llama3.2", # Ensure this model is available in your Ollama setup
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant that provides extremely concise answers.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.1, 'top_p': 0.5}
            )
            return query, response['message']['content'].strip()
        except ollama.ResponseError as e:
            print(f"Ollama Response Error for query '{query}': {e.error}") # .error often has more specific info
            return query, f"Ollama Error: {e.error}"
        except Exception as e:
            print(f"An unexpected error occurred for query '{query}': {e}")
            return query, f"Unexpected Error: {e}"

async def main():
    tsv_file = "data/trimmed.tsv"
    output_csv_file = "data/test.csv"
    string_col_name = "string_data" # Column in TSV to extract queries from
    
    pandas_read_chunk_size = 10000  # How many rows pandas reads from TSV at a time
    api_call_batch_size = 500     # Number of queries to process in one async batch (yield_batch_size for extractor)
    max_concurrent_requests = 10  # Max parallel calls to Ollama

    print("\n___Generating the Expanded Queries___\n")

    # Define the expected header for the output CSV
    csv_header = "query,expanded"
    num_already_expanded = await count_csv_data_rows(output_csv_file, header_line_content=csv_header)
    
    if num_already_expanded > 0:
        print(f"Resuming: Found {num_already_expanded} already expanded queries in {output_csv_file}.")
        open_mode = 'a' # Append mode
        write_header = False
    else:
        print(f"Starting fresh or empty output file. Writing to {output_csv_file}.")
        open_mode = 'w' # Write mode (will overwrite or create new)
        write_header = True
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)


    async with aiofiles.open(output_csv_file, mode=open_mode, newline='', encoding='utf-8') as file:
        if write_header:
            await file.write(csv_header + '\n')

        semaphore = asyncio.Semaphore(max_concurrent_requests)
        api_batches_processed_count = 0

        # The extractor will skip the first 'num_already_expanded' valid queries from the TSV.
        data_extractor = extract_string_batches(
            tsv_filepath=tsv_file,
            string_column_name=string_col_name,
            pandas_chunk_size=pandas_read_chunk_size,
            yield_batch_size=api_call_batch_size,
            skip_n_values=0
        )

        for batch_of_queries_to_expand in data_extractor:
            if not batch_of_queries_to_expand: # If extractor yields an empty list
                continue

            tasks = []
            for query_to_expand in batch_of_queries_to_expand:
                if not isinstance(query_to_expand, str) or not query_to_expand.strip():
                    # print(f"Skipping empty or invalid query: '{query_to_expand}'")
                    continue # Skip empty or non-string queries before sending to API
                tasks.append(expand_query(query=query_to_expand, semaphore=semaphore))
            
            # if tasks:
            #     results = await tqdm.gather(*tasks, desc=f"Processing API Batch {api_batches_processed_count + 1}/{len(batch_of_queries_to_expand)} tasks")
                
            #     lines_to_write = []
            #     for original_query, expanded_query in results:
            #         # Basic CSV escaping for quotes
            #         oq_escaped = original_query.replace('"', '""')
            #         eq_escaped = expanded_query.replace('"', '""')
            #         lines_to_write.append(f'"{oq_escaped}","{eq_escaped}"\n')
                
            #     await file.write("".join(lines_to_write))
            #     await file.flush() # Ensure data is written to disk after each batch
            
            api_batches_processed_count += 1
            # Optional: Limit the number of API batches for testing
            # if api_batches_processed_count >= 2:
            #     print("--- Testing limit: Reached 2 API batches. ---")
            #     break
    
    print(f"\nProcessing complete. Expanded queries saved to {output_csv_file}.")

if __name__ == "__main__":
    
    asyncio.run(main())