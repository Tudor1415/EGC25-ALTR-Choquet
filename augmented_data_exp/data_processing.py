# data_processing.py
import os
import pandas as pd

def add_rule_columns(df, rules_df):
    # Get the max value in the entire DataFrame
    last_item = df.max().max()
    
    # Process each of the top 10 rules
    for index, row in rules_df.head(10).iterrows(): # Only top 10 rules
        rule = row['Rule']
        rule_index = index + 1
        column_name = f"Rule_{rule_index}"
        
        # Parse rule to separate antecedent and consequent elements
        antecedent, consequent = rule.split(' => ')
        antecedent_elements = [int(x.strip()) for x in antecedent.strip('[]').split(';')]
        consequent_elements = [int(x.strip()) for x in consequent.strip('[]').split(';')]
        
        # Combine all elements that need to be checked for the rule to be valid
        all_rule_elements = antecedent_elements + consequent_elements
        
        # Apply rule validity check (all antecedent items are in the transaction) on each row
        df[column_name] = df.apply(
            lambda x: last_item + rule_index 
            if all(elem in x.values for elem in all_rule_elements) 
            else last_item + rule_index + 1,
            axis=1
        )
    
    return df

def process_datasets(dat_file_folder, output_base):
    # Iterate over each dataset directory
    for dataset_name in os.listdir(dat_file_folder):
        dataset_name = dataset_name.split(".")[0]
        rules_path = os.path.join(output_base, dataset_name, "samples")
        # Check if the dataset directory has any files
        if os.path.isdir(rules_path):
            # Find the first CSV file in the samples folder
            first_file = next((f for f in os.listdir(rules_path) if f.endswith('.csv')), None)
            if first_file:
                rules_file_path = os.path.join(rules_path, first_file)
                try:
                    # Read the rules data from the first CSV file
                    rules_df = pd.read_csv(rules_file_path)

                    # Add rule columns to this dataset
                    df = pd.read_csv(dat_file_folder + dataset_name + ".dat", delimiter=' ')
                    df = add_rule_columns(df, rules_df)

                    aug_directory = os.path.join(output_base, dataset_name, "aug")

                    os.makedirs(aug_directory, exist_ok=True)

                    output_path = os.path.join(aug_directory, dataset_name + "_processed.dat")

                    df.to_csv(output_path, sep=' ', index=False, header=False)
                                    
                    print(f"Processed {dataset_name} with new rule columns.")
                    
                except Exception as e:
                    print(f"Failed to process {rules_file_path}: {e}")
