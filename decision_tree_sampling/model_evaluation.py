import os
import numpy as np
import pandas as pd

# Assuming these modules are provided elsewhere in your project
from measures import information_gain, phi
from data_processing import extract_decision_tree_rules, get_rule_stats, compute_jaccard_distance_matrix


# --------------------------------------
# Data Loading Functions
# --------------------------------------

def load_dataset(dataset_file):
    """
    Load dataset from a .dat file.

    Args:
        dataset_file (str): Path to the .dat dataset file.

    Returns:
        tuple: A tuple containing the feature matrix X and target vector y.
    """
    df = pd.read_csv(dataset_file, sep=" ", header=None)
    X = df.iloc[:, :-1]  # Features (all columns except the last one)
    y = df.iloc[:, -1]   # Target variable (the last column)
    return X, y


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
        csv_files = [
            f for f in os.listdir(sample_path)
            if f.endswith('.csv') and ((keyword.lower() in f.lower()) != exclude)
        ]

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
    Load a distance matrix from a CSV file and convert it to a Numpy array.

    Args:
        filepath (str): Full path to the CSV file containing the distance matrix.

    Returns:
        np.ndarray: A 2D Numpy array representing the distance matrix.
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


# --------------------------------------
# Evaluation Functions
# --------------------------------------

def compute_rule_measure(rule_stats, measure, n):
    """
    Compute the measure value for a single rule.

    Args:
        rule_stats (dict): Statistics for the rule.
        measure (str): The measure to compute ('IG' or 'phi').
        n (int): Total number of samples.

    Returns:
        float: The computed measure value.
    """
    nx0 = rule_stats['nx0']
    nx1 = rule_stats['nx1']
    n11 = rule_stats['n11']
    n00 = rule_stats['n00']
    n01 = rule_stats['n01']
    n0x = rule_stats['n0x']
    n1x = rule_stats['n1x']

    if measure == 'IG':
        value = information_gain(nx0, nx1, n11, n00, n01, n0x, n)
    elif measure == 'phi':
        value = phi(n, n11, n1x, nx1, n0x, nx0)
    else:
        raise ValueError("Invalid measure. Choose either 'IG' or 'phi'.")

    return value


def evaluate_tree_on_dataset(X, y, class_values, measure, top_k, depth_range=10, repetitions=5):
    """
    Evaluate decision tree rules on the dataset, compute the mean of the top_k measure values,
    and compute the Jaccard distance matrix for the rules corresponding to these top_k values.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        class_values (list): List of class values.
        measure (str): The measure to compute ('IG' or 'phi').
        depth_range (int): The maximum depth of the tree.
        repetitions (int): Number of repetitions for tree extraction.
        top_k (int): Number of top measure values to consider.

    Returns:
        float: Mean of the top_k measure values.
        np.ndarray: Jaccard distance matrix for top_k rules.
    """
    measure_values_with_covers = []
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

                    # Compute the measure
                    n = len(X)
                    rule_stats = stats[rule]
                    value = compute_rule_measure(rule_stats, measure, n)
                    measure_values_with_covers.append((value, rule_stats['cover']))

    # If not enough values were collected
    if len(measure_values_with_covers) < top_k:
        raise ValueError(f"Not enough scores computed, {len(measure_values_with_covers)} < {top_k}. Increase depth_range or repetitions, or reduce top_k.")

    # Select the top k measure values and corresponding covers
    top_k_values_with_covers = sorted(measure_values_with_covers, key=lambda x: x[0], reverse=True)[:top_k]
    top_k_values, top_k_covers = zip(*top_k_values_with_covers)

    # Compute the Jaccard distance matrix for the top k covers
    distance_matrix = compute_jaccard_distance_matrix(list(top_k_covers))

    return compute_redundancy_score(top_k_values, distance_matrix)


def compute_redundancy_score(top_k_values, distance_matrix):
    """
    Compute the redundancy score using the top_k_values and distance_matrix.

    Args:
        top_k_values (pd.Series or list): Series of top_k measure values.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        float: Redundancy score.
    """
    score = top_k_values[0]

    for i in range(1, len(top_k_values)):
        min_distance = min([distance_matrix[i][k] for k in range(0, i)])
        score += min_distance * top_k_values[i]

    return score


def evaluate_datasets(dat_files_folder, output_base, measure, top_k):
    """
    Evaluate datasets by computing the mean of top_k measure values and redundancy score.

    Args:
        dat_files_folder (str): Directory containing .dat dataset files.
        output_base (str): Base directory for output files.
        measure (str): The measure to compute ('IG' or 'phi').
        top_k (int): Number of top measure values to consider.

    Returns:
        dict: Dictionary of dataset names and their corresponding scores.
    """
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
            print(f"Processing the top {top_k} samples on dataset {dataset_name}...")

            # Load dataset
            dataset_path = os.path.join(dat_files_folder, dataset_file)
            X, y = load_dataset(dataset_path)

            if len(np.unique(y)) < 2:
                print(f"Not enough classes in original dataset {dataset_name}, skipping.")
                continue

            # Evaluate tree on dataset
            class_values = class_items_dict.get(dataset_name, None)
            if class_values is None:
                print(f"No class values found for dataset {dataset_name}, skipping.")
                continue

            tree_score = evaluate_tree_on_dataset(
                X, y, class_values, measure, top_k
            )

            # Load sampled rules and distance matrix
            sample_path = os.path.join(output_base, dataset_name, 'samples', measure)
            sampled_rules_path = fetch_first_csv(sample_path, keyword='jaccard', exclude=True)
            if sampled_rules_path is None:
                continue

            sampled_rules = pd.read_csv(sampled_rules_path)
            distance_matrix_path = fetch_first_csv(sample_path)
            distance_matrix = load_distance_matrix(distance_matrix_path)
            if distance_matrix is None:
                continue

            top_k_sample = sampled_rules[measure].head(top_k).values
            sample_score = compute_redundancy_score(top_k_sample, distance_matrix)

            results[dataset_name] = (tree_score, sample_score)

    return results
