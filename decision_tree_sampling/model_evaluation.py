import os
import numpy as np
import pandas as pd
from measures import information_gain, phi
from data_processing import extract_decision_tree_rules, get_rule_stats

def evaluate_tree_on_dataset(X, y, depth_range=10, repetitions=5, measure='IG'):
    max_values = []

    for depth in range(1, depth_range + 1):
        for _ in range(repetitions):
            # Extract decision tree rules (assuming `extract_decision_tree_rules` exists)
            rules = extract_decision_tree_rules(X, y, max_depth=depth)

            # Get rule statistics (assuming `get_rule_stats` exists)
            stats = get_rule_stats(X, y, rules)

            measure_values = []
            for rule in stats:
                # Extract statistics for current rule
                nx0 = stats[rule]['nx0']
                nx1 = stats[rule]['nx1']
                n11 = stats[rule]['n11']
                n00 = stats[rule]['n00']
                n01 = stats[rule]['n01']
                n0x = stats[rule]['n0x']
                n1x = stats[rule]['n1x']
                
                # Get total number of samples
                n = len(X)

                # Compute the score based on the chosen measure
                if measure == 'IG':
                    value = information_gain(nx0, nx1, n11, n00, n01, n0x, n)
                elif measure == 'phi':
                    # Compute the phi score using `phi()` function
                    value = phi(n, n11, n1x, nx1, n0x, nx0)
                else:
                    raise ValueError("Invalid measure. Choose either 'IG' or 'phi'.")

                measure_values.append(value)

            # Append the maximum value for the current depth and repetition
            max_values.append(max(measure_values))

    # Return the overall maximum measure value
    return max(max_values)


def load_dataset(dataset_file):
    df = pd.read_csv(dataset_file, sep=" ", header=None)  
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target variable (the last column)

    return X, y

def evaluate_datasets(dat_files_folder, output_base, measure):
    results = {}
    
    for dataset_file in os.listdir(dat_files_folder):
        if dataset_file.endswith('.dat'):
            dataset_name = os.path.splitext(dataset_file)[0]

            print(f"Processing dataset {dataset_name}...")

            # Load dataset
            dataset_path = os.path.join(dat_files_folder, dataset_file)
            X, y = load_dataset(dataset_path)

            if len(np.unique(y)) < 2:
                print(f"Not enough classes in original dataset {dataset_name}, skipping.")
                continue
            
            best_score_tree = evaluate_tree_on_dataset(X, y, measure=measure)
            
            sample_path = os.path.join(output_base, dataset_name, 'samples', measure)
            
            csv_files = [f for f in os.listdir(sample_path) if f.endswith('.csv')]

            if csv_files:
                first_csv_file = csv_files[0]
                first_csv_path = os.path.join(sample_path, first_csv_file)
                sampled_rules = pd.read_csv(first_csv_path)
            else:
                print(f"No CSV files found in directory: {sample_path}")
            
            best_score_sample = sampled_rules[measure].max()
            
            results[dataset_name] = (best_score_tree, best_score_sample)
    
    return results