# data_processing.py
import os
import pandas as pd
import re
import operator

def add_rule_columns_discretized(df, rules_df):
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

def add_rule_columns(df, rules_df, mapping):
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
            lambda row: 1 
            if all(row[mapping[elem][0]] for elem in antecedent_elements 
                  if mapping[elem][1](mapping[elem][0], row[mapping[elem][0]])) 
            else 0,
            axis=1
        )
    
    return df

def process_dat_files(dat_file_folder, output_base):
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
                    df = add_rule_columns_discretized(df, rules_df)

                    aug_directory = os.path.join(output_base, dataset_name, "aug")

                    os.makedirs(aug_directory, exist_ok=True)

                    output_path = os.path.join(aug_directory, dataset_name + "_processed_disc.dat")
                    print(df.head())
                    df.to_csv(output_path, sep=' ', index=False, header=False)
                                    
                    print(f"Processed {dataset_name} with new rule columns.")
                    
                except Exception as e:
                    print(f"Failed to process {rules_file_path}: {e}")

def process_datasets(dataset_names, dataset_folder, output_base, mapping_folder):
    # Iterate over each dataset directory
    for dataset_name in dataset_names:
        print()
        mapping_file_path = os.path.join(mapping_folder, dataset_name + ".map")
        mapping = create_conditions_dict(mapping_file_path)
        
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
                    df = pd.read_csv(dataset_folder + dataset_name + "csv", delimiter=';')
                    df = add_rule_columns(df, rules_df, mapping)
                    print(df.head())
                    aug_directory = os.path.join(output_base, dataset_name, "aug")

                    os.makedirs(aug_directory, exist_ok=True)

                    output_path = os.path.join(aug_directory, dataset_name + "_processed_non_disc.dat")

                    df.to_csv(output_path, sep=' ', index=False, header=False)
                                    
                    print(f"Processed {dataset_name} with new rule columns.")
                    
                except Exception as e:
                    print(f"Failed to process {rules_file_path}: {e}")


def create_conditions_dict(mapping_file_path):
    map_df = pd.read_csv(mapping_file_path, sep="\t")
    conditions_dict = {}
    
    # Define operator mappings
    operators = {
        '<': operator.lt,
        '<=': operator.le,
        '=': operator.eq,
        '>=': operator.ge,
        '>': operator.gt
    }
    
    # Process each row in the DataFrame
    for _, row in map_df.iterrows():
        encoding = int(row['Encoding'])
        condition_str = row['Column_Name'].strip()

        # Match patterns like 'variable < value', 'variable >= value', 'variable = value', 'variable = (value1 - value2)'
        match = re.match(r'(\w[\w\-]*)\s*(<|<=|=|>=|>)\s*(.+)', condition_str)
        if match:
            variable, op_str, value = match.groups()
            op_func = operators[op_str.strip()]
            value = value.strip().replace('(', '').replace(')', '')

            # Check for range conditions (e.g., "Amount_of_given_credit_card = (45000 - 145000)")
            if ' - ' in value and op_str == '=':
                low, high = map(float, value.split('-'))
                conditions_dict[encoding] = (variable.strip(), lambda var_name, var_value, var=variable, low=low, high=high: var_name == var and low <= float(var_value) <= high)
            else:
                # Handle simple conditions with numeric or categorical values
                val = float(value) if value.replace('.', '', 1).isdigit() else value
                conditions_dict[encoding] = (variable.strip(), lambda var_name, var_value, var=variable, op=op_func, val=val: var_name == var and op(float(var_value), val) if isinstance(val, (int, float)) else var_name == var and op(var_value, val))
        else:
            # Handle conditions without operators (e.g., "Gender = 1")
            if '=' in condition_str:
                variable, value = condition_str.split('=')
                value = value.strip().replace('(', '').replace(')', '')
                conditions_dict[encoding] = (variable.strip(), lambda var_name, var_value, var=variable.strip(), val=value: var_name == var and str(var_value) == val)
            else:
                # Handle direct categorical values without '=' (e.g., "class_0" and "class_1")
                variable = condition_str.strip()
                conditions_dict[encoding] = (variable.strip(), lambda var_name, var_value, var=variable: var_name == var and str(var_value) == '1')
    
    return conditions_dict