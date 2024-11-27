import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from scipy.stats import entropy

def regroup_files_by_metadata(directory_path):
    grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for filename in os.listdir(directory_path):
        if not (filename.endswith('.json') or "_input" in filename):
            continue
        try:
            filename = filename.split("_input")[0]
            dataset_name, fold_idx, algorithm_name, oracle_name_with_ext = filename.split('_')
            fold_idx = int(fold_idx.replace('fold', ''))
        except ValueError:
            print(f"Skipping invalid filename format: {filename}")
            continue
        file_path = os.path.join(directory_path, filename + "_input.json")
        grouped_files[dataset_name][oracle_name][algorithm_name][fold_idx] = file_path
    return {dataset: {oracle: dict(algo) for oracle, algo in algos.items()} for dataset, algos in grouped_files.items()}

def load_and_aggregate_data(file_dict):
    all_data = []
    for file_path in file_dict.values():
        with open(file_path, 'r') as f:
            data = json.load(f)
        all_data.append(data)
    return all_data

def compute_preference_distances(all_data, p=2):
    distances = []
    for data in all_data:
        preferences = data["preferences"]
        alternatives = data["alternatives"]
        for pref in preferences:
            index1, index2, _ = pref
            alt1 = alternatives[index1 - 1]  # Adjust if indices are 0-based, assuming 1-based
            alt2 = alternatives[index2 - 1]
            distance = minkowski(alt1, alt2, p)
            distances.append(distance)
    return np.array(distances)

def min_max_normalize(data):
    min_val, max_val = np.min(data), np.max(data)
    return (data - min_val) / (max_val - min_val) if max_val > min_val else data

def plot_cdf(data, labels, title="CDF Plot", xlabel="Distance", ylabel="CDF", output_dir=None, filename="cdf_plot.png"):
    plt.figure(figsize=(10, 5))
    for dist, label in zip(data, labels):
        sorted_data = np.sort(dist)
        cdf = np.cumsum(sorted_data) / np.sum(sorted_data)
        plt.plot(sorted_data, cdf, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()

def compute_domination_vectors_and_histogram(algorithm_files, output_dir=None, p=2):
    """
    Compute domination vectors for a given algorithm and dataset, plot histogram, and compute entropy.
    """
    domination_counts = defaultdict(int)
    for fold_idx, file_path in algorithm_files.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
        preferences = data["preferences"]
        alternatives = data["alternatives"]

        for pref in preferences:
            index1, index2, _ = pref
            alt1 = alternatives[index1 - 1]  # Adjust for 1-based index
            alt2 = alternatives[index2 - 1]

            domination_vector = []
            for v1, v2 in zip(alt1, alt2):
                if v1 > v2:
                    domination_vector.append(1)
                elif v1 < v2:
                    domination_vector.append(-1)
                else:
                    domination_vector.append(0)

            # Convert to a tuple to use as a dictionary key
            domination_counts[tuple(domination_vector)] += 1

    # Convert domination counts to probabilities for entropy calculation
    domination_values = np.array(list(domination_counts.values()))
    domination_probabilities = domination_values / np.sum(domination_values)

    # Compute entropy
    dom_entropy = entropy(domination_probabilities)

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.bar(
        [' '.join(map(str, key)) for key in domination_counts.keys()],
        domination_counts.values()
    )
    plt.title("Domination Vector Histogram")
    plt.xlabel("Domination Vector")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.grid(axis='y')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "domination_histogram.png")
        plt.savefig(save_path)
        print(f"Saved histogram to {save_path}")
    plt.show()

    print(f"Entropy of normalized domination distribution: {dom_entropy:.4f}")
    print(f"Number of distinct domination vectors: {len(domination_counts)}")

    return domination_counts, dom_entropy

def compute_normalized_domination_vectors_and_histogram(algorithm_files, output_dir=None, p=2):
    """
    Compute normalized domination vectors (treating vector and -vector as equivalent) for a given algorithm and dataset,
    plot histogram, and compute entropy.
    """
    domination_counts = defaultdict(int)

    for fold_idx, file_path in algorithm_files.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
        preferences = data["preferences"]
        alternatives = data["alternatives"]

        for pref in preferences:
            index1, index2, _ = pref
            alt1 = alternatives[index1 - 1]  # Adjust for 1-based index
            alt2 = alternatives[index2 - 1]

            domination_vector = []
            for v1, v2 in zip(alt1, alt2):
                if v1 > v2:
                    domination_vector.append(1)
                elif v1 < v2:
                    domination_vector.append(-1)
                else:
                    domination_vector.append(0)

            # Normalize domination vector by choosing the lexicographically larger between the vector and its negation
            domination_tuple = tuple(domination_vector)
            negation_tuple = tuple(-x for x in domination_vector)
            canonical_vector = max(domination_tuple, negation_tuple)  # Pick lexicographically larger tuple

            domination_counts[canonical_vector] += 1

    # Convert domination counts to probabilities for entropy calculation
    domination_values = np.array(list(domination_counts.values()))
    domination_probabilities = domination_values / np.sum(domination_values)

    # Compute entropy
    dom_entropy = entropy(domination_probabilities)

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.bar(
        [' '.join(map(str, key)) for key in domination_counts.keys()],
        domination_counts.values()
    )
    plt.title("Normalized Domination Vector Histogram")
    plt.xlabel("Normalized Domination Vector")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.grid(axis='y')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "normalized_domination_histogram.png")
        plt.savefig(save_path)
        print(f"Saved histogram to {save_path}")
    plt.show()

    print(f"Entropy of normalized domination distribution: {dom_entropy:.4f}")
    print(f"Number of distinct domination vectors: {len(domination_counts)}")   
    return domination_counts, dom_entropy

# Example usage
directory_path = "results/ExpMaximumEntropy/input"
dataset_name, oracle_name, algorithm_name = "bank", "chiSquared", "KappalabIterative-MaximumEntropySampling"
print(dataset_name, oracle_name, algorithm_name)
# directory_path = "results_ablation/BradleyTerry_ChangePeriod_10_MaxIter_250/input"
# dataset_name, oracle_name, algorithm_name = "bank", "InformationGain", "KappalabIterative-UncertaintySampling"

output_directory_path = os.path.join(directory_path, "output_plots")
grouped_files = regroup_files_by_metadata(directory_path)


# algorithm_files = grouped_files[dataset_name][oracle_name]
# distances = []
# labels = []
# for algorithm, files in algorithm_files.items():
#     all_data = load_and_aggregate_data(files)
#     dist = compute_preference_distances(all_data)
#     distances.append(dist)
#     labels.append(algorithm)

# plot_cdf(distances, labels, title=f"All Algorithms Distance CDF ({dataset_name}, {oracle_name})", output_dir=output_directory_path, filename=f"{dataset_name}_{oracle_name}_all_algorithms_cdf.png")
algorithm_files = grouped_files[dataset_name][oracle_name][algorithm_name]
domination_counts, dom_entropy = compute_normalized_domination_vectors_and_histogram(algorithm_files, output_dir=output_directory_path)