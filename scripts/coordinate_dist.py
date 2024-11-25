import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def regroup_files_by_metadata(directory_path):
    grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for filename in os.listdir(directory_path):
        if not filename.endswith('.json'):
            continue
        try:
            dataset_name, fold_idx, algorithm_name, oracle_name_with_ext = filename.split('_')
            oracle_name = oracle_name_with_ext.split('.json')[0]
            fold_idx = int(fold_idx.replace('fold', ''))
        except ValueError:
            print(f"Skipping invalid filename format: {filename}")
            continue
        file_path = os.path.join(directory_path, filename)
        grouped_files[dataset_name][oracle_name][algorithm_name][fold_idx] = file_path
    return {dataset: {oracle: dict(algo) for oracle, algo in algos.items()} for dataset, algos in grouped_files.items()}

def load_data(grouped_files, dataset_name, algorithm_name, oracle_name):
    all_data = []
    try:
        for fold_idx, file_path in grouped_files[dataset_name][oracle_name][algorithm_name].items():
            with open(file_path, 'r') as file:
                data = json.load(file)
                all_data.append(data['alternatives'])
    except KeyError:
        print(f"No data found for {algorithm_name} with oracle {oracle_name} in dataset {dataset_name}")
    return all_data

def calculate_statistics(data_by_oracle):
    for oracle, data in data_by_oracle.items():
        # Flatten the list of lists to get a single list of all alternatives across all folds
        num_alternatives = [len(fold) for fold in data]
        mean_num = np.mean(num_alternatives)
        std_num = np.std(num_alternatives)
        print(f"{oracle}: Mean = {mean_num:.2f}, Std = {std_num:.2f}")

# Example usage
directory_path = "results_ab/ExpAllDatasets/input"
grouped_files = regroup_files_by_metadata(directory_path)

datasets = ["bank", "credit"]
algorithm_name = "KappalabIterative-Mining"
oracles = ["InformationGain", "chiSquared"]

for dataset_name in datasets:
    print(f"\nDataset: {dataset_name}")
    data_by_oracle = {oracle: load_data(grouped_files, dataset_name, algorithm_name, oracle) for oracle in oracles}
    calculate_statistics(data_by_oracle)
