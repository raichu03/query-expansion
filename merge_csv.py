import csv
import sys

csv.field_size_limit(sys.maxsize)

def merge_csv_files(file1_path, file2_path, output_path):
    """
    Merge two CSV files with the same headers into a single output file.
    
    Args:
        file1_path (str): Path to the first input CSV file
        file2_path (str): Path to the second input CSV file
        output_path (str): Path to the output merged CSV file
    """
    try:
        # Open both input files in read mode
        with open(file1_path, 'r', newline='', encoding='utf-8') as f1, \
             open(file2_path, 'r', newline='', encoding='utf-8') as f2:
            
            # Create CSV readers for both files
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            
            # Read the headers (first row) from both files
            headers1 = next(reader1)
            headers2 = next(reader2)
            
            # Verify that headers match
            if headers1 != headers2:
                print("Error: CSV files have different headers")
                print(f"File 1 headers: {headers1}")
                print(f"File 2 headers: {headers2}")
                sys.exit(1)
            
            # Open the output file in write mode
            with open(output_path, 'w', newline='', encoding='utf-8') as out_file:
                writer = csv.writer(out_file)
                
                # Write the header row
                writer.writerow(headers1)
                
                # Write all rows from first file
                for row in reader1:
                    writer.writerow(row)
                
                # Write all rows from second file
                for row in reader2:
                    writer.writerow(row)
        
        print(f"Successfully merged files. Output saved to {output_path}")
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    
    # Uncomment to use command line arguments instead
    if len(sys.argv) != 4:
        print("Usage: python merge_csv.py <file1> <file2> <output>")
        sys.exit(1)
    file1, file2, output = sys.argv[1], sys.argv[2], sys.argv[3]
    
    merge_csv_files(file1, file2, output)