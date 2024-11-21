import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from measures import information_gain, phi
from data_processing import extract_random_forest_rules, get_rule_stats, compute_jaccard_distance_matrix

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

def evaluate_tree_on_dataset(X, y, class_values, measure, top_k, depth_range=10, repetitions=5):
    """
    Evaluate decision tree rules on the dataset, compute the redundancy score,
    and return the top_k measure values and covers.

    Returns:
        tuple: redundancy_score, top_k_values, top_k_covers, actual_k
    """
    measure_values_with_covers = []
    seen_rules = set()
    total_iterations = depth_range * repetitions

    bar_format = '{desc}: |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]'

    with tqdm(total=total_iterations, desc="Processing", unit="iteration",
              bar_format=bar_format, ncols=80, ascii='=>') as pbar:
        for depth in range(1, depth_range + 1):
            for _ in range(repetitions):
                # Extract decision tree rules
                rules = extract_random_forest_rules(X, y, max_depth=depth)
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
                        cover = rule_stats['cover']
                        measure_values_with_covers.append((value, cover))
                # Update the progress bar
                pbar.update(1)

    # Adjust top_k to be the minimum of desired top_k and number of rules collected
    actual_k = min(top_k, len(measure_values_with_covers))
    if actual_k == 0:
        raise ValueError(f"No rules extracted for dataset. Cannot evaluate.")

    # Select the top actual_k measure values and corresponding covers
    top_k_values_with_covers = sorted(measure_values_with_covers, key=lambda x: x[0], reverse=True)[:actual_k]
    top_k_values, top_k_covers = zip(*top_k_values_with_covers)

    # Compute the Jaccard distance matrix for the top k covers
    distance_matrix = compute_jaccard_distance_matrix(list(top_k_covers))

    redundancy_score = compute_redundancy_score(top_k_values, distance_matrix)

    return redundancy_score, top_k_values, top_k_covers, actual_k

# Modified function
def evaluate_datasets(dat_files_folder, output_base, measure, top_k):
    """
    Evaluate datasets by computing redundancy scores and collecting rule data.

    Returns:
        tuple: results, tree_data_per_dataset, sample_data_per_dataset
    """
    class_items_dict = {
        "adult": [145, 146],
        "bank": [89, 90],
        "connect": [127, 128],
        "credit": [111, 112],
        "dota": [346, 347],
        "toms": [911, 912],
        "mushroom": [116, 117],
        "banknote": [17, 18],
        "heart": [32, 33],
        "ionosphere": [145, 146],
        "ilpd": [15, 16],
        "magic": [80, 81],
        "medical_kaggle": [126, 127],
        "parkinsons": [52, 53],
        "pima": [31, 32],
        "skin": [120, 121],
        "tictactoe": [28, 29],
        "transfusion": [7, 8],
        "travel-insurance": [212, 213],
        "twitter": [1512, 1513],
        "wdbc": [89, 90],
        "weatherAUS": [152, 153],
        "iris": [12, 13]
    }

    results = {}
    tree_data_per_dataset = {}
    sample_data_per_dataset = {}

    for dataset_file in os.listdir(dat_files_folder):
        if dataset_file.endswith('.dat'):
            dataset_name = os.path.splitext(dataset_file)[0]
            print(f"Processing dataset {dataset_name}...")

            # Load dataset
            dataset_path = os.path.join(dat_files_folder, dataset_file)
            X, y = load_dataset(dataset_path)

            if len(np.unique(y)) < 2:
                print(f"Not enough classes in dataset {dataset_name}, skipping.")
                continue

            # Evaluate tree on dataset
            class_values = class_items_dict.get(dataset_name, None)
            if class_values is None:
                print(f"No class values found for dataset {dataset_name}, skipping.")
                continue

            try:
                tree_score, top_k_values, top_k_covers, actual_k = evaluate_tree_on_dataset(
                    X, y, class_values, measure, top_k
                )
            except ValueError as e:
                print(str(e))
                continue

            # Adjust top_k for sampling method as well
            adjusted_k = actual_k  # Number of rules actually used

            # Store the extracted data for tree method
            tree_data_per_dataset[dataset_name] = {
                'values': top_k_values,
                'covers': top_k_covers,
                'distance_matrix': None,  # Will compute when needed
                'mean_distances': None,
                'actual_k': adjusted_k
            }

            # Load sampled rules and distance matrix
            sample_path = os.path.join(output_base, dataset_name, 'samples', measure)
            sampled_rules_path = fetch_first_csv(sample_path, keyword='jaccard', exclude=True)
            if sampled_rules_path is None:
                print(f"No sampled rules CSV found for dataset {dataset_name}, skipping.")
                continue

            sampled_rules = pd.read_csv(sampled_rules_path)
            distance_matrix_path = fetch_first_csv(sample_path)
            distance_matrix = load_distance_matrix(distance_matrix_path)
            if distance_matrix is None:
                print(f"No distance matrix found for dataset {dataset_name}, skipping.")
                continue

            # Adjust top_k if necessary
            if len(sampled_rules) < adjusted_k:
                adjusted_k = len(sampled_rules)
                print(f"Adjusted top_k to {adjusted_k} for dataset {dataset_name} due to insufficient sampled rules.")

            top_k_sample = sampled_rules[measure].head(adjusted_k).values
            sample_distance_matrix = distance_matrix[:adjusted_k, :adjusted_k]
            sample_score = compute_redundancy_score(top_k_sample, sample_distance_matrix)

            # Store the extracted data for sample method
            sample_data_per_dataset[dataset_name] = {
                'values': top_k_sample,
                'distance_matrix': sample_distance_matrix,
                'mean_distances': None,
                'actual_k': adjusted_k
            }

            results[dataset_name] = (tree_score, sample_score)

    return results, tree_data_per_dataset, sample_data_per_dataset

def plot_distance_cdfs(tree_data_per_dataset, sample_data_per_dataset, output_base, measure):
    """
    Plot two CDFs on the same plot, one for each method (sampling and tree), using the entries of the distance matrices.
    """
    distances_tree = []
    distances_sample = []

    for dataset_name in tree_data_per_dataset.keys():
        tree_data = tree_data_per_dataset[dataset_name]
        sample_data = sample_data_per_dataset.get(dataset_name)

        if not sample_data:
            continue  # Skip if sample data is not available

        # Get the distance matrix for tree method
        if tree_data['distance_matrix'] is None:
            # Compute distance matrix
            distance_matrix = compute_jaccard_distance_matrix(list(tree_data['covers']))
            tree_data['distance_matrix'] = distance_matrix
        else:
            distance_matrix = tree_data['distance_matrix']

        # Adjust adjusted_k to be the minimum between actual_k and the size of the distance matrix
        adjusted_k = min(tree_data['actual_k'], distance_matrix.shape[0])

        # Flatten the upper triangle of the distance matrix, excluding the diagonal
        triu_indices = np.triu_indices(adjusted_k, k=1)
        distances = distance_matrix[triu_indices]
        distances_tree.extend(distances)

        # Do the same for sample data
        if sample_data['distance_matrix'] is None:
            distance_matrix = compute_jaccard_distance_matrix(list(sample_data['covers']))
            sample_data['distance_matrix'] = distance_matrix
        else:
            distance_matrix = sample_data['distance_matrix']

        # Adjust adjusted_k for sample data
        adjusted_k = min(sample_data['actual_k'], distance_matrix.shape[0])

        # Flatten the upper triangle of the distance matrix, excluding the diagonal
        triu_indices = np.triu_indices(adjusted_k, k=1)
        distances = distance_matrix[triu_indices]
        distances_sample.extend(distances)

    # Ensure that we have data to plot
    if not distances_tree or not distances_sample:
        print("No data available to plot.")
        return

    # Get the number of data points for each method
    num_points_tree = len(distances_tree)
    num_points_sample = len(distances_sample)

    # Compute CDFs
    sorted_tree_distances = np.sort(distances_tree)
    sorted_sample_distances = np.sort(distances_sample)

    cdf_tree = np.arange(1, len(sorted_tree_distances)+1) / len(sorted_tree_distances)
    cdf_sample = np.arange(1, len(sorted_sample_distances)+1) / len(sorted_sample_distances)

    # Plot the CDFs
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_tree_distances, cdf_tree, label='Tree Method')
    plt.plot(sorted_sample_distances, cdf_sample, label='Sampling Method')

    plt.xlabel('Distance Between Rules')
    plt.ylabel('Cumulative Distribution Function (CDF)')
    plt.title(f'CDF of Distances for {measure} Across Datasets\n'
              f'Number of Data Points - Tree: {num_points_tree}, Sampling: {num_points_sample}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_filename = f'distance_cdf_{measure}.png'
    plot_path = os.path.join(output_base, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    print(f"CDF plot saved at {plot_path}.")