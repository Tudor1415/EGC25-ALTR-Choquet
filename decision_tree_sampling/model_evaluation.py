import os
import numpy as np
import pandas as pd
from measures import information_gain
from data_processing import extract_decision_tree_rules, get_rule_stats

def evaluate_tree_on_dataset(X, y, depth_range = 10, repetitions = 5):
    max_values = []

    for depth in range(1, depth_range+1):
        for _ in range(repetitions):
            rules = extract_decision_tree_rules(X, y, max_depth = depth)

            stats = get_rule_stats(X, y, rules)

            information_gain_values = []
            for rule in stats:
                information_gain_values.append(information_gain(stats[rule]['nx0'], stats[rule]['nx1'], stats[rule]['n11'], stats[rule]['n00'], stats[rule]['n01'], stats[rule]['n0x'], len(X)))
            
            max_values.append(max(information_gain_values))

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
            
            best_score_tree = evaluate_tree_on_dataset(X, y)
            
            sample_path = os.path.join(output_base, dataset_name, 'samples', measure)
            
            sampled_rules = pd.read_csv(sample_path)
            
            best_score_sample = sampled_rules[measure].max()
            
            results[dataset_name] = (best_score_tree, best_score_sample)
    
    return results