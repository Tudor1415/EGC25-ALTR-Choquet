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

def compute_redundancy_score(top_k_sample, distance_matrix) :
    score = top_k_sample[0]
    
    for i in range(1, len(top_k_sample)):
        min_distance = min([distance_matrix[i][k] for k in range(0, i)])
        score += min_distance * top_k_sample[i]

    return score

def fetch_first_csv(sample_path, keyword='jaccard', exclude=False):
    """
    Fetch the first CSV file from a directory that contains or excludes a specific keyword in its filename.

    Args:
    sample_path (str): The directory path where the files are located.
    keyword (str): Keyword to include or exclude in the file names.
    exclude (bool): If True, exclude files containing the keyword; if False, include them.

    Returns:
    str: The full path to the first matched CSV file or None if no file is found.
    """
    try:
        # List all CSV files and filter them based on the presence of the keyword
        csv_files = [f for f in os.listdir(sample_path)
                     if f.endswith('.csv') and ((keyword.lower() in f.lower()) != exclude)]

        # Check if any suitable files were found
        if csv_files:
            first_csv_file = csv_files[0]
            first_csv_path = os.path.join(sample_path, first_csv_file)
            return first_csv_path 
        else:
            print(f"No suitable CSV files found in directory: {sample_path}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def load_distance_matrix(filepath):
    """
    Load a Jaccard distance matrix from a CSV file and convert it to a Numpy matrix.

    Args:
    filepath (str): Full path to the CSV file containing the distance matrix.

    Returns:
    np.ndarray: A 2D Numpy array representing the Jaccard distance matrix.
    """
    if filepath is None:
        return None

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath, header=None)  # Assuming no header in the distance matrix CSV

        # Convert the DataFrame to a Numpy array
        matrix = df.values

        return matrix
    except Exception as e:
        print(f"An error occurred while loading the distance matrix: {e}")
        return None
    
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
            
            top_k_tree = evaluate_tree_on_dataset(X, y, class_items_dict[dataset_name], measure, top_k = top_k)

            sample_path = os.path.join(output_base, dataset_name, 'samples', measure)
            sampled_rules_path = fetch_first_csv(sample_path, keyword='jaccard', exclude=True)
            sampled_rules = pd.read_csv(sampled_rules_path)
            distance_matrix_path = fetch_first_csv(sample_path)
            distance_matrix = load_distance_matrix(distance_matrix_path)
            top_k_sample = sampled_rules[measure].head(top_k)
            
            score = compute_redundancy_score(top_k_sample, distance_matrix)

            
            results[dataset_name] = (top_k_tree, score)
    
    return results