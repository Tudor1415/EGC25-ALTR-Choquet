import os
import subprocess
import argparse
import sys
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run ExtractSample Java program on datasets.')
parser.add_argument('--nb_samples', type=int, default=100_000, help='Number of samples to generate')
args = parser.parse_args()

nb_samples = args.nb_samples

# Paths relative to the scripts folder
scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(scripts_dir, '..'))
dataset_folder = os.path.join(project_root, 'data', 'dat-files') + "\\"
target_dir = os.path.join(project_root, 'target')
class_output_dir = os.path.join(target_dir, 'classes')
output_base = os.path.join(project_root, 'results', 'augmented_data_exp')

# Timeout in minutes
timeout_minutes = 30

# Measure names and weights (can be adjusted as needed)
measure_names = 'phi'
weights = '1'

# Step 1: Get the classpath of the project including dependencies
print("Getting classpath from Maven...")
mvn_dependency_command = ['mvn', 'dependency:build-classpath', '-Dmdep.outputFile=classpath.txt']
# Run this instead: mvn dependency:build-classpath -Dmdep.outputFile=classpath.txt  

# Read the classpath from the generated file
classpath_file = os.path.join(project_root, 'classpath.txt')
if not os.path.exists(classpath_file):
    print("Failed to generate classpath using Maven.")
    sys.exit(1)

with open(classpath_file, 'r') as f:
    maven_classpath = f.read().strip()

# Combine the Maven classpath with the compiled classes in target/classes
execution_classpath = os.pathsep.join([class_output_dir, maven_classpath])

# Step 2: Iterate over all dataset files in the dataset folder
for dataset_file in os.listdir(dataset_folder):
    if os.path.isfile(os.path.join(dataset_folder, dataset_file)):
        dataset_name = dataset_file

        # Construct the output directory
        dataset_name_without_extension = os.path.splitext(dataset_name)[0]
        output_directory = os.path.join(output_base, dataset_name_without_extension, 'samples')

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Construct the command to run the Java program
        command = [
            'java',
            '-cp',
            execution_classpath,
            'tools.utils.sampling.ExtractSample',
            dataset_name,
            dataset_folder,
            output_directory,
            str(nb_samples),
            str(timeout_minutes),
            measure_names,
            weights
        ]

        print(f"Processing dataset {dataset_name} with {nb_samples} samples...")
        # Call the Java program
        try:
            subprocess.run(command, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {dataset_name}: {e}")
        except FileNotFoundError:
            print("Java executable not found. Please ensure Java is installed and added to your system's PATH.")
            sys.exit(1)

import pandas as pd
# Function to add columns for each rule based on the top ten extracted rules
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
        
        # Apply rule validity check on each row
        df[column_name] = df.apply(
            lambda x: last_item + rule_index 
            if all(elem in x.values for elem in all_rule_elements) 
            else last_item + rule_index + 1,
            axis=1
        )
    
    return df

# Iterate over each dataset directory
for dataset_name in os.listdir(dataset_folder):
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
                # Assuming `rules_df` contains the rules with columns like 'Rule', 'ScoreApprox', etc.
                print(f"Processing rules from {rules_file_path}")
                
                # Add rule columns to this dataset
                df = pd.read_csv(dataset_folder + dataset_name + ".dat", delimiter=' ')
                df = add_rule_columns(df, rules_df)

                aug_directory = os.path.join(output_base, dataset_name, "aug")

                os.makedirs(aug_directory, exist_ok=True)

                output_path = os.path.join(aug_directory, dataset_name + "_processed.dat")

                df.to_csv(output_path, sep=' ', index=False, header=False)
                                
                print(f"Processed {rules_file_path} with new rule columns:")
                
            except Exception as e:
                print(f"Failed to process {rules_file_path}: {e}")