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

def plot_cumulative_distributions(grouped_files, dataset_name, oracle_name, p=2, output_dir=None):
    algorithms = grouped_files[dataset_name][oracle_name]
    all_scores = []
    all_distances = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))  # Color map

    # Collect data from each algorithm
    for algo_name, files in algorithms.items():
        scores, alternatives = load_and_aggregate_data(files)
        all_scores.append(np.array(scores))
        distances = [minkowski(alt1, alt2, p) for alt1, alt2 in combinations(alternatives, 2)]
        all_distances.append(np.array(distances))

    # Setup the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Dataset: {dataset_name}, Oracle: {oracle_name}")

    # Plot Score Cumulative Distributions
    for i, algo_name in enumerate(algorithms.keys()):
        sorted_scores = np.sort(all_scores[i])
        cdf = np.cumsum(sorted_scores) / np.sum(sorted_scores)
        axes[0].plot(sorted_scores, cdf / cdf[-1], label=f"{algo_name}", color=colors[i])  # Normalize by the last value

    # Plot Distance Cumulative Distributions
    for i, algo_name in enumerate(algorithms.keys()):
        sorted_distances = np.sort(all_distances[i])
        cdf = np.cumsum(sorted_distances) / np.sum(sorted_distances)
        axes[1].plot(sorted_distances, cdf / cdf[-1], label=f"{algo_name}", color=colors[i])  # Normalize by the last value

    axes[0].set_title("Score Cumulative Distribution")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Cumulative Density")
    axes[0].grid(True)

    axes[1].set_title("Distance Cumulative Distribution")
    axes[1].set_xlabel(f"L{p} Distance")
    axes[1].set_ylabel("Cumulative Density")
    axes[1].grid(True)

    # Place a single legend below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Increase bottom margin
    # Save the plot if an output directory is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{dataset_name}_{oracle_name}_cumulative_plots.png")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    # plt.show()

# Example usage
# Example usage
directory_path = "results_ab/ExpAllDatasets/input"
output_directory_path = os.path.join(directory_path, "output_plots")
grouped_files = regroup_files_by_metadata(directory_path)
plot_cumulative_distributions(grouped_files, "bank", "chiSquared", p=2, output_dir=output_directory_path)
plot_cumulative_distributions(grouped_files, "bank", "InformationGain", p=2, output_dir=output_directory_path)
plot_cumulative_distributions(grouped_files, "credit", "chiSquared", p=2, output_dir=output_directory_path)
plot_cumulative_distributions(grouped_files, "credit", "InformationGain", p=2, output_dir=output_directory_path)