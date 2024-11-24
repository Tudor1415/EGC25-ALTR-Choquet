import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import minkowski

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

def load_and_aggregate_data(file_dict):
    all_scores = []
    all_alternatives = []
    for file_path in file_dict.values():
        with open(file_path, 'r') as f:
            data = json.load(f)
        all_scores.extend([score[0] for score in data['scoreMap'].values()])
        all_alternatives.extend(data['alternatives'])
    return all_scores, all_alternatives

def compute_global_cdf(scores, distances, p=2):
    # Combine all scores and distances from each algorithm
    combined_scores = np.hstack(scores)
    combined_distances = np.hstack(distances)

    # Compute global CDF for scores
    sorted_scores = np.sort(combined_scores)
    score_cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    # Compute global CDF for distances
    sorted_distances = np.sort(combined_distances)
    distance_cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)

    return sorted_scores, score_cdf, sorted_distances, distance_cdf

def plot_cdfs(grouped_files, dataset_name, oracle_name, p=2):
    algorithms = grouped_files[dataset_name][oracle_name]
    all_scores = []
    all_distances = []

    # Collect data from each algorithm
    for algo_name, files in algorithms.items():
        scores, alternatives = load_and_aggregate_data(files)
        all_scores.append(np.array(scores))
        distances = [minkowski(alt1, alt2, p) for alt1, alt2 in combinations(alternatives, 2)]
        all_distances.append(np.array(distances))

    # Compute global CDFs
    sorted_scores, score_cdf, sorted_distances, distance_cdf = compute_global_cdf(all_scores, all_distances, p)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Dataset: {dataset_name}, Oracle: {oracle_name}")

    # Plot Score CDFs for each algorithm
    for i, algo_name in enumerate(algorithms.keys()):
        local_scores_sorted = np.sort(all_scores[i])
        local_score_cdf = np.searchsorted(sorted_scores, local_scores_sorted, side="right") / len(sorted_scores)
        axes[0].plot(local_scores_sorted, local_score_cdf, label=f"{algo_name} Score CDF")

    # Plot Distance CDFs for each algorithm
    for i, algo_name in enumerate(algorithms.keys()):
        local_distances_sorted = np.sort(all_distances[i])
        local_distance_cdf = np.searchsorted(sorted_distances, local_distances_sorted, side="right") / len(sorted_distances)
        axes[1].plot(local_distances_sorted, local_distance_cdf, label=f"{algo_name} Distance CDF")

    axes[0].set_title("Score CDF")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Cumulative Probability")
    axes[0].legend()

    axes[1].set_title("Distance CDF")
    axes[1].set_xlabel(f"L{p} Distance")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplots to fit the figure title
    plt.show()

# Example usage
directory_path = "results/ExpAllDatasets/input"
grouped_files = regroup_files_by_metadata(directory_path)
plot_cdfs(grouped_files, "bank", "InformationGain", p=2)
