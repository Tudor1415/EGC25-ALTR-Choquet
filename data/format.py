import pandas as pd

def write_list_of_lists_to_file(data, output_file):
    with open(output_file, 'w') as f:
        for inner_list in data:
            f.write(' '.join(map(str, inner_list)) + '\n')

def reformat(input_file, output_file, mapping_file):
    # Load the CSV file into Python
    data = pd.read_csv(input_file, sep=';')

    # Create new columns based on distinct values in the "class" column
    distinct_values = data['class'].unique()
    for value in distinct_values:
        column_name = f'class_{value}'
        data[column_name] = (data['class'] == value).astype(int)

    # Remove the original "class" column
    data.drop(columns=['class'], inplace=True)
    
    # Multiply each column by its respective column index
    for i, col in enumerate(data.columns):
        data[col] *= (i + 1)

    # Remove zeros from the rows and keep only non-zero values
    non_zero_rows = []
    for _, row in data.iterrows():
        non_zero_rows.append([item for item in row if item != 0])

    # Write column index -> other words mapping to a text file
    mapping_df = pd.DataFrame({'Encoding': range(1, len(data.columns) + 1), 'Column_Name': data.columns})
    mapping_df.to_csv(mapping_file, sep='\t', index=False)

    # Write the reformatted data to a new DAT file
    write_list_of_lists_to_file(non_zero_rows, output_file)

# Retrieve command line arguments
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 4:
    sys.exit("Usage: python format.py <input_csv_file> <output_dat_file> <mapping_file>")

# Extract arguments
input_csv_file = sys.argv[1]
output_dat_file = sys.argv[2]
mapping_file = sys.argv[3]

# Call the function to convert CSV to DAT
reformat(input_csv_file, output_dat_file, mapping_file)
