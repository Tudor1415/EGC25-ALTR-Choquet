import os
import subprocess
import argparse
import sys
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run ExtractSample Java program on datasets.')
parser.add_argument('--nb_samples', type=int, default=1000, help='Number of samples to generate')
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
measure_names = 'confidence,support'
weights = '0.5,0.5'

# Ensure the output directory exists
os.makedirs(output_base, exist_ok=True)

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
    print(df.head())    
    # Process each of the top ten rules in rules_df
    for index, row in rules_df.head(10).iterrows():  # Only top 10 rules
        rule = row['Rule']
        rule_index = index + 1  # 1-based indexing for rule
        
        # Create a new column for each rule and populate it with max_value + rule_index if rule is valid
        column_name = f"Rule_{rule_index}"
        df[column_name] = df.apply(
            lambda x: max_value + rule_index if rule in str(x['Rule']) else None, axis=1
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
            print(first_file, rules_file_path)
            try:
                # Read the rules data from the first CSV file
                rules_df = pd.read_csv(rules_file_path)
                
                # Assuming `rules_df` contains the rules with columns like 'Rule', 'ScoreApprox', etc.
                print(f"Processing rules from {rules_file_path}")
                
                # Add rule columns to this dataset
                df = pd.read_csv(dataset_folder + dataset_name, delimiter=' ')
                df = add_rule_columns(df, rules_df)
                
                # Print the resulting DataFrame with new rule columns (for verification)
                print(f"Processed {rules_file_path} with new rule columns:")
                print(df.head())
                
            except Exception as e:
                print(f"Failed to process {rules_file_path}: {e}")