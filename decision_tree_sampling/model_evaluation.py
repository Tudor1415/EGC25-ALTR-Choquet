import os
import numpy as np
import pandas as pd
from measures import information_gain, phi
from data_processing import extract_decision_tree_rules, get_rule_stats

def evaluate_tree_on_dataset(X, y, class_values, measure, depth_range=10, repetitions=5, top_k=10):
    all_measure_values = []
    seen_rules = set() 

    for depth in range(1, depth_range + 1):
        for _ in range(repetitions):
            # Extract decision tree rules 
            rules = extract_decision_tree_rules(X, y, max_depth=depth)

            # Get rule statistics 
            stats = get_rule_stats(X, y, rules, class_values)

            for rule in stats:
                rule_hash = hash(tuple(rule))

                if rule_hash not in seen_rules:
                    seen_rules.add(rule_hash)

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
                        value = phi(n, n11, n1x, nx1, n0x, nx0)
                    else:
                        raise ValueError("Invalid measure. Choose either 'IG' or 'phi'.")

                    all_measure_values.append(value)

    if len(all_measure_values) < top_k:
        raise ValueError("Not enough scores computed. Increase depth_range or repetitions, or reduce top_k.")

    top_k_values = sorted(all_measure_values, reverse=True)[:top_k]
    mean_top_k = np.mean(top_k_values)

    return mean_top_k

def load_dataset(dataset_file):
    df = pd.read_csv(dataset_file, sep=" ", header=None)  
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target variable (the last column)

    return X, y

def evaluate_datasets(dat_files_folder, output_base, measure, top_k = 10):
    class_items_dict = {
        'adult': [145, 146],
        'bank': [89, 90],
        'connect': [127, 128],
        'credit': [111, 112],
        'dota': [346, 347],
        'toms': [911, 912],
        'mushroom': [116, 117]
    }

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
            
            top_k_mean_tree = evaluate_tree_on_dataset(X, y, class_items_dict[dataset_name], measure, top_k = top_k)

            sample_path = os.path.join(output_base, dataset_name, 'samples', measure)
            
            csv_files = [f for f in os.listdir(sample_path) if f.endswith('.csv')]

            if csv_files:
                first_csv_file = csv_files[0]
                first_csv_path = os.path.join(sample_path, first_csv_file)
                sampled_rules = pd.read_csv(first_csv_path)
            else:
                print(f"No CSV files found in directory: {sample_path}")
            
            sorted_sample = sampled_rules.sort_values(by=measure, ascending=False)
            top_k_mean_sample = sorted_sample[measure].head(top_k).mean()

            
            results[dataset_name] = (top_k_mean_tree, top_k_mean_sample)
    
    return results