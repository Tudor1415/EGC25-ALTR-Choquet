import os
import subprocess
import argparse
import sys
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run ExtractSample Java program on datasets.')
parser.add_argument('--nb_samples', type=int, default=100_000, help='Number of samples to generate')
args = parser.parse_args()

nb_samples = args.nb_samples
nb_folds = 5

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
        
        # Apply rule validity check (all antecedent items are in the transaction) on each row
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

                # Add rule columns to this dataset
                df = pd.read_csv(dataset_folder + dataset_name + ".dat", delimiter=' ')
                df = add_rule_columns(df, rules_df)

                aug_directory = os.path.join(output_base, dataset_name, "aug")

                os.makedirs(aug_directory, exist_ok=True)

                output_path = os.path.join(aug_directory, dataset_name + "_processed.dat")

                df.to_csv(output_path, sep=' ', index=False, header=False)
                                
                print(f"Processed {dataset_name} with new rule columns.")
                
            except Exception as e:
                print(f"Failed to process {rules_file_path}: {e}")

# Define the class items for each dataset
class_items_dict = {
    'adult': ['145', '146'],
    'bank': ['89', '90'],
    'connect': ['127', '128'],
    'credit': ['111', '112'],
    'dota': ['346', '347'],
    'toms': ['911', '912'],
    'mushroom': ['116', '117']
}

# Function to load dataset and return feature matrix X and labels y
def load_dataset(dataset_file, class_items):
    data = []
    with open(dataset_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            data.append([int(item) for item in items])

    # Collect all items
    all_items = set()
    for transaction in data:
        all_items.update(transaction)

    # Map items to columns
    item_to_col = {item: idx for idx, item in enumerate(sorted(all_items))}

    n_samples = len(data)
    n_features = len(all_items)
    X = np.zeros((n_samples, n_features), dtype=int)
    y = np.full(n_samples, -1)

    for i, transaction in enumerate(data):
        items_in_transaction = set(transaction)
        for item in items_in_transaction:
            col_idx = item_to_col[item]
            X[i, col_idx] = 1

        # Assign labels
        for label, class_item in enumerate(class_items):
            if class_item in items_in_transaction:
                y[i] = label
                break

    # Remove samples without labels
    valid_indices = y != -1
    X = X[valid_indices]
    y = y[valid_indices]

    return X, y

# Collect results
results = {}

for dataset_file in os.listdir(dataset_folder):
    if dataset_file.endswith('.dat'):
        dataset_name = os.path.splitext(dataset_file)[0]
        if dataset_name not in class_items_dict:
            print(f"Class items not defined for dataset {dataset_name}, skipping.")
            continue

        print(f"Processing dataset {dataset_name}...")

        # Read class items
        class_items = [int(item) for item in class_items_dict[dataset_name]]

        # Load original dataset
        dataset_path = os.path.join(dataset_folder, dataset_file)
        X_orig, y_orig = load_dataset(dataset_path, class_items)

        # Load augmented dataset
        aug_dataset_file = os.path.join(output_base, dataset_name, "aug", dataset_name + "_processed.dat")
        if not os.path.exists(aug_dataset_file):
            print(f"Augmented dataset not found for {dataset_name}, skipping augmented dataset.")
            X_aug, y_aug = None, None
        else:
            X_aug, y_aug = load_dataset(aug_dataset_file, class_items)

        # Function to evaluate model using K-fold cross-validation
        def evaluate_model(X, y, n_splits=nb_folds):
            accuracies = []
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Train Random Forest
                clf = RandomForestClassifier(random_state=42)
                clf.fit(X_train, y_train)
                # Predict on test set
                y_pred = clf.predict(X_test)
                # Compute accuracy
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)
            return accuracies

        # Evaluate on original dataset
        if len(np.unique(y_orig)) < 2:
            print(f"Not enough classes in original dataset {dataset_name}, skipping.")
            continue
        acc_orig = evaluate_model(X_orig, y_orig)

        # Evaluate on augmented dataset if available
        if X_aug is not None and y_aug is not None:
            if len(np.unique(y_aug)) < 2:
                print(f"Not enough classes in augmented dataset {dataset_name}, skipping augmented dataset.")
                acc_aug = None
            else:
                acc_aug = evaluate_model(X_aug, y_aug)
        else:
            acc_aug = None

        # Store results
        results[dataset_name] = {'Original': acc_orig, 'Augmented': acc_aug}

plot_data = []

for dataset_name in results.keys():
    acc_orig = results[dataset_name]['Original']
    acc_aug = results[dataset_name]['Augmented']

    # Skip datasets without valid results
    if not acc_orig and not acc_aug:
        print(f"No valid results for dataset {dataset_name}, skipping.")
        continue

    # Add original accuracies to the plot data
    if acc_orig:
        for acc in acc_orig:
            plot_data.append({'Dataset_Type': f'{dataset_name}_Original', 'Accuracy': acc})

    # Add augmented accuracies to the plot data
    if acc_aug:
        for acc in acc_aug:
            plot_data.append({'Dataset_Type': f'{dataset_name}_Augmented', 'Accuracy': acc})

# Convert the plot data to a DataFrame
df_plot = pd.DataFrame(plot_data)

# Prepare data for matplotlib boxplot
data_to_plot = []
labels = []

for dataset_type in df_plot['Dataset_Type'].unique():
    data_to_plot.append(df_plot[df_plot['Dataset_Type'] == dataset_type]['Accuracy'].values)
    labels.append(dataset_type)

# Create the figure
plt.figure(figsize=(12, 6))

# Create the boxplot
bp = plt.boxplot(
    data_to_plot,
    labels=labels,
    patch_artist=True,
    showmeans=True,
    meanprops={
        "marker": "D",
        "markerfacecolor": "orange",
        "markeredgecolor": "black"
    }
)

# Set properties for each box
for box in bp['boxes']:
    # Change outline color
    box.set(color='black', linewidth=1)
    # Change fill color with alpha
    box.set(facecolor=(1, 1, 1, 0.3))  # White with alpha=0.3

# Set color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=1)

# Set color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='black', linewidth=1)

# Set color and linewidth of the medians
for median in bp['medians']:
    median.set(color='black', linewidth=1)

# Set style for the means
for mean in bp['means']:
    mean.set(marker='D', markerfacecolor='orange', markeredgecolor='black')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Set the title and labels
plt.title('Accuracy Comparison Across Datasets')
plt.ylabel('Accuracy')
plt.xlabel('Dataset_Type')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_base, f"all_datasets_accuracy_boxplot.png"))
plt.close()
print("Combined plot saved for all datasets.")
print("Processing complete.")