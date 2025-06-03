import sys

def remove_rows_up_to(input_file, output_file, remove_up_to_row):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile):
            if i > remove_up_to_row:  # Keep rows after the specified row
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python remove_tsv_rows.py <input_file.tsv> <output_file.tsv> <remove_up_to_row>")
        print("Example: python remove_tsv_rows.py input.tsv output.tsv 5  (removes rows 0-5, keeps from row 6 onwards)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        remove_up_to_row = int(sys.argv[3])
    except ValueError:
        print("Error: remove_up_to_row must be an integer.")
        sys.exit(1)
    
    remove_rows_up_to(input_file, output_file, remove_up_to_row)
    print(f"Removed rows 0-{remove_up_to_row} from '{input_file}'. Kept remaining rows in '{output_file}'.")