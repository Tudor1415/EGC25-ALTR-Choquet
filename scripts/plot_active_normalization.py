#!/usr/bin/env python3

import os
import sys
import re
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import kendalltau

def parse_filename(filename):
    """
    Parses the filename to extract datasetName, foldID, LearningAlgorithm, normMethod, Oracle, and timestamp.
    Expected filename format: datasetName_foldID_LearningAlgorithm-normMethod_Oracle_timestamp.csv
    """
    basename = os.path.basename(filename)
    basename = basename[:-4]  # Remove the '.csv' extension
    parts = basename.split('_')

    if len(parts) < 5:
        print(f"Filename {filename} does not match expected pattern.")
        return None

    datasetName = parts[0]
    foldID = parts[1]
    oracle = parts[-2]
    timestamp = parts[-1]

    # Join the middle parts back together in case LearningAlgorithm or normMethod contain underscores
    middle_part = '_'.join(parts[2:-2])

    # Split LearningAlgorithm and normMethod at the last hyphen
    if '-' in middle_part:
        # Split from the right in case there are hyphens in LearningAlgorithm
        learning_algo, norm_method = middle_part.rsplit('-', 1)
    else:
        learning_algo = middle_part
        norm_method = ''

    return datasetName, foldID, learning_algo, norm_method, oracle, timestamp

def group_files(directory):
    """
    Groups files by datasetName, normMethod, Oracle, LearningAlgorithm, and foldID in a nested dictionary.
    """
    files_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))

    # List all files in the directory
    print("Grouping files...")
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            parsed = parse_filename(filename)
            if parsed:
                datasetName, foldID, LearningAlgorithm, normMethod, Oracle, timestamp = parsed
                # Organize files by datasetName, normMethod, Oracle, LearningAlgorithm, and foldID
                file_path = os.path.join(directory, filename)
                files_dict[datasetName][normMethod][Oracle][LearningAlgorithm][foldID].append(file_path)
                print(f"Grouped file: {file_path}")
            else:
                print(f"Filename {filename} does not match expected pattern.")
    return files_dict

def compute_kendalltau_correlation(df):
    """
    Computes the kendalltau correlation coefficient between rankApprox and rankOracle.
    """
    correlation, _ = kendalltau(df['rankApprox'], df['rankOracle'])
    return correlation

def compute_rankings(df):
    """
    Computes rankings based on scoreApprox and scoreOracle.
    Adds 'rankApprox' and 'rankOracle' columns to the DataFrame.
    """
    df = df.copy()
    df['rankApprox'] = df['scoreApprox'].rank(ascending=False, method='first')
    df['rankOracle'] = df['scoreOracle'].rank(ascending=False, method='first')
    return df

def compute_average_precision_at_k(df, k):
    """
    Computes Average Precision at top k entries.
    """
    df_top_k = df.nsmallest(k, 'rankApprox')
    relevant = df_top_k['rankOracle'] <= k
    precision_at_k = relevant.sum() / k
    return precision_at_k

def compute_unique_ratio(df):
    """
    Computes the percentage of unique values in df["scoreOracle"].
    """
    unique_values = df["scoreOracle"].nunique()
    total_values = len(df["scoreOracle"])
    unique_ratio = (unique_values / total_values) * 100
    return unique_ratio

def process_files(grouped_files):
    """
    Processes each group of files, computes metrics, and organizes results by datasetName, then normMethod, then Oracle, then LearningAlgorithm, and foldID.
    The metrics are stored in order based on the ascending timestamp extracted from the file names.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'avg_precision': [], 'kendalltau': [], 'timestamps': []})))))

    top_percentage = 0.1

    print("Processing files...")
    for datasetName, normMethods in grouped_files.items():
        print(f"Dataset: {datasetName}")
        for normMethod, oracles in normMethods.items():
            print(f"  Norm Method: {normMethod}")
            for oracle, algos in oracles.items():
                print(f"    Oracle: {oracle}")
                for algo, folds in algos.items():
                    print(f"      Algorithm: {algo}")
                    for foldID, files in folds.items():
                        print(f"        Fold ID: {foldID}")
                        # Temporary storage to sort by timestamp before final storage
                        temp_storage = []

                        for file in files:
                            # print(f"          Processing file: {file}")
                            df = pd.read_csv(file)
                            if 'scoreApprox' not in df.columns or 'scoreOracle' not in df.columns:
                                print(f"          Warning: 'scoreApprox' or 'scoreOracle' columns not found in {file}. Skipping file.")
                                continue
                            df = compute_rankings(df)
                            k = max(1, int(len(df) * top_percentage))

                            avg_precision = compute_average_precision_at_k(df, k)
                            kendalltau_corr = compute_kendalltau_correlation(df)

                            parsed = parse_filename(file)
                            if parsed:
                                _, _, _, _, _, timestamp = parsed
                                temp_storage.append((int(timestamp), avg_precision, kendalltau_corr))
                            else:
                                print(f"          Filename {file} does not match expected pattern.")

                        # Sort by timestamp and store the sorted data
                        temp_storage.sort()
                        for timestamp, avg_prec, kendalltau in temp_storage:
                            results[datasetName][normMethod][oracle][algo][foldID]['timestamps'].append(timestamp)
                            results[datasetName][normMethod][oracle][algo][foldID]['avg_precision'].append(avg_prec)
                            results[datasetName][normMethod][oracle][algo][foldID]['kendalltau'].append(kendalltau)
                        
                        result = results[datasetName][normMethod][oracle][algo][foldID]
                        result['avg_precision_cumm'] = list(np.cumsum(result['avg_precision']))
                        result['kendalltau_cumm'] = list(np.cumsum(result['kendalltau']))

    return results

def collect_all_algorithms(results):
    """
    Collects all unique algorithm names from the results.
    """
    algorithms_set = set()
    for datasetName, normMethods in results.items():
        for normMethod, oracles in normMethods.items():
            for oracle, algorithms in oracles.items():
                for algo in algorithms.keys():
                    algorithms_set.add(algo)
    return sorted(algorithms_set)

def create_algorithm_color_mapping(algorithms):
    """
    Creates a consistent color mapping for algorithms.
    """
    num_algorithms = len(algorithms)
    color_palette = sns.color_palette("deep", num_algorithms)
    algorithm_color_mapping = dict(zip(algorithms, color_palette))
    return algorithm_color_mapping

def plot_metrics(results, algorithm_color_mapping):
    """
    Generates and saves plots for average precision and kendalltau correlation from the results data,
    plotting metrics for each algorithm on the same graph with different colors and legend.

    Parameters:
        results (dict): The dictionary containing the nested results from process_files.
        algorithm_color_mapping (dict): A dictionary mapping algorithms to specific colors.
    """
    print("Plotting metrics...")
    for datasetName, normMethods in results.items():
        for normMethod, oracles in normMethods.items():
            for oracle, algorithms in oracles.items():
                print(f"Creating plots for Dataset: {datasetName}, Norm Method: {normMethod}, Oracle: {oracle}")
                # Initialize lists to collect data for each algorithm
                data_precision = []
                data_kendalltau = []

                for algo, folds_data in algorithms.items():
                    print(f"  Processing Algorithm: {algo}")
                    # Initialize dictionaries to collect data for each metric across all iterations
                    precision_data_algo = defaultdict(list)
                    kendalltau_data_algo = defaultdict(list)

                    # Collect data across folds for each iteration
                    for foldID, metrics in folds_data.items():
                        print(f"    Processing Fold ID: {foldID}")
                        for i, (prec, spear) in enumerate(zip(metrics['avg_precision_cumm'], metrics['kendalltau_cumm'])):
                            precision_data_algo[i].append(prec)
                            kendalltau_data_algo[i].append(spear)

                    # Accumulate data from all iterations for this algorithm
                    for iteration, prec_values in precision_data_algo.items():
                        for prec in prec_values:
                            data_precision.append({
                                'Algorithm': algo,
                                'Iteration': iteration + 1,
                                'Average Precision': prec
                            })

                    for iteration, spear_values in kendalltau_data_algo.items():
                        for spear in spear_values:
                            data_kendalltau.append({
                                'Algorithm': algo,
                                'Iteration': iteration + 1,
                                'kendalltau Correlation': spear
                            })

                if not data_precision or not data_kendalltau:
                    print(f"No data available for Dataset: {datasetName}, Norm Method: {normMethod}, Oracle: {oracle}. Skipping plot.")
                    continue  # Skip if no data

                # After collecting data for all algorithms, create DataFrames
                df_precision = pd.DataFrame(data_precision)
                df_kendalltau = pd.DataFrame(data_kendalltau)

                # Now, create a figure with two subplots side by side
                fig, axes = plt.subplots(1, 2, figsize=(20, 10))

                # Plot Average Precision
                sns.lineplot(
                    ax=axes[0],
                    x='Iteration',
                    y='Average Precision',
                    hue='Algorithm',
                    data=df_precision,
                    palette=algorithm_color_mapping,
                    marker='o'
                )
                axes[0].set_title(f"{datasetName} - {oracle} - {normMethod} - Average Precision")
                axes[0].set_xlabel('Iteration Number')
                axes[0].set_ylabel('Average Precision')
                axes[0].set_xticks(range(1, df_precision['Iteration'].max() + 1, max(1, df_precision['Iteration'].max() // 10)))
                axes[0].legend(title='Algorithm')

                # Plot kendalltau Correlation
                sns.lineplot(
                    ax=axes[1],
                    x='Iteration',
                    y='kendalltau Correlation',
                    hue='Algorithm',
                    data=df_kendalltau,
                    palette=algorithm_color_mapping,
                    marker='o'
                )
                axes[1].set_title(f"{datasetName} - {oracle} - {normMethod} - kendalltau Correlation")
                axes[1].set_xlabel('Iteration Number')
                axes[1].set_ylabel('kendalltau Correlation')
                axes[1].set_xticks(range(1, df_kendalltau['Iteration'].max() + 1, max(1, df_kendalltau['Iteration'].max() // 10)))
                axes[1].legend(title='Algorithm')

                # Adjust layout and save figure
                plt.tight_layout()
                os.makedirs("results/active_normalization/output_plots/", exist_ok=True)
                output_filename = "results/active_normalization/output_plots/" + f"{datasetName}_{oracle}_{normMethod}.pdf"
                plt.savefig(output_filename, format='pdf')
                plt.close()
                print(f"Plot saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description='Process CSV files to compute metrics and generate plots.')
    parser.add_argument('directory', type=str, help='Directory containing CSV files.')
    args = parser.parse_args()

    data_dir = args.directory
    if not os.path.isdir(data_dir):
        print(f"The directory {data_dir} does not exist or is not a directory.")
        sys.exit(1)

    grouped_files = group_files(data_dir)
    if not grouped_files:
        print("No files were grouped. Please check if the directory contains CSV files with the correct naming pattern.")
        sys.exit(1)

    results = process_files(grouped_files)
    if not results:
        print("No results were generated. Please check if the CSV files contain the required data.")
        sys.exit(1)

    # Collect all unique algorithms
    all_algorithms = collect_all_algorithms(results)
    print(f"All algorithms: {all_algorithms}")

    # Create a consistent color mapping for algorithms
    algorithm_color_mapping = create_algorithm_color_mapping(all_algorithms)
    print(f"Algorithm color mapping: {algorithm_color_mapping}")

    plot_metrics(results, algorithm_color_mapping)

if __name__ == "__main__":
    main()
