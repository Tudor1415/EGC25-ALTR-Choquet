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
from scipy.stats import spearmanr
from scipy.stats import kendalltau

def parse_filename(filename):
    """
    Parses the filename to extract datasetName, foldID, LearningAlgorithm, Oracle, and timestamp.
    Expected filename format: datasetName_foldID_LearningAlgorithm_Oracle_timestamp.csv
    """
    basename = os.path.basename(filename)
    pattern = r'^(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv$'
    match = re.match(pattern, basename)
    if match:
        datasetName, foldID, LearningAlgorithm, Oracle, timestamp = match.groups()
        return datasetName, foldID, LearningAlgorithm, Oracle, timestamp
    else:
        return None

def group_files(directory):
    """
    Groups files by datasetName, LearningAlgorithm, Oracle, and foldID in a nested dictionary.
    """
    files_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            parsed = parse_filename(filename)
            if parsed:
                datasetName, foldID, LearningAlgorithm, Oracle, timestamp = parsed
                # Organize files by datasetName, then LearningAlgorithm, then Oracle, and foldID
                file_path = os.path.join(directory, filename)
                files_dict[datasetName][Oracle][LearningAlgorithm][foldID].append(file_path)
            else:
                print(f"Filename {filename} does not match expected pattern.")
    return files_dict

def compute_spearman_correlation(df):
    """
    Computes the Spearman correlation coefficient between rankApprox and rankOracle.
    """
    correlation, _ = spearmanr(df['rankApprox'], df['rankOracle'])
    return correlation

def compute_rankings(df):
    """
    Computes rankings based on scoreApprox and scoreOracle.
    Returns two Series containing the rankings.
    """
    df = df.copy()
    df['rankApprox'] = df['scoreApprox'].rank(ascending=False, method='first')
    df['rankOracle'] = df['scoreOracle'].rank(ascending=False, method='first')
    return df

def compute_average_precision_at_k(df, k):
    """
    Computes Average Precision at top k entries.
    """
    df = df.nsmallest(k, 'rankApprox')
    relevant = df['rankOracle'] <= k
    precision_at_k = relevant.sum() / k
    return precision_at_k

def compute_regret_at_k(df, k):
    """
    Computes Regret at top k entries.
    """
    top_k_approx = set(df.nsmallest(k, 'rankApprox').index)
    top_k_oracle = set(df.nsmallest(k, 'rankOracle').index)
    regret = len(top_k_oracle - top_k_approx) / k
    return regret

def compute_unique_ratio(df, k):
    """
    Computes the percentage of unique values in df["scoreOracle"].
    """
    unique_values = df["scoreOracle"].nunique()  
    total_values = len(df["scoreOracle"]) 
    
    unique_ratio = (unique_values / total_values) * 100  
    
    return unique_ratio

def compute_kendalltau_correlation(df):
    """
    Computes the kendalltau correlation coefficient between rankApprox and rankOracle.
    """
    correlation, _ = kendalltau(df['rankApprox'], df['rankOracle'])
    return correlation

def process_files(grouped_files, cumulative):
    """
    Processes each group of files, computes metrics, and organizes results by datasetName, then normMethod, then Oracle, then LearningAlgorithm, and foldID.
    The metrics are stored in order based on the ascending timestamp extracted from the file names.
    
    If cumulative is True, cumulative sums are calculated for average precision.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'avg_precision_1': [], 'avg_precision_10': [], 'timestamps': []}))))

    top_percentage_1 = 0.01
    top_percentage_10 = 0.10

    print("Processing files...")
    for datasetName, oracles in grouped_files.items():
        print(f"Dataset: {datasetName}")
        for oracle, algos in oracles.items():
            print(f"    Oracle: {oracle}")
            for algo, folds in algos.items():
                print(f"      Algorithm: {algo}")
                for foldID, files in folds.items():
                    print(f"        Fold ID: {foldID}")
                    # Temporary storage to sort by timestamp before final storage
                    temp_storage = []

                    for file in files:
                        df = pd.read_csv(file)
                        if 'scoreApprox' not in df.columns or 'scoreOracle' not in df.columns:
                            print(f"          Warning: 'scoreApprox' or 'scoreOracle' columns not found in {file}. Skipping file.")
                            continue
                        df = compute_rankings(df)
                        
                        k_1 = max(1, int(len(df) * top_percentage_1))
                        k_10 = max(1, int(len(df) * top_percentage_10))

                        avg_precision_1 = compute_average_precision_at_k(df, k_1)
                        avg_precision_10 = compute_average_precision_at_k(df, k_10)

                        parsed = parse_filename(file)
                        if parsed:
                            _, _, _, _, timestamp = parsed
                            temp_storage.append((int(timestamp), avg_precision_1, avg_precision_10))
                        else:
                            print(f"          Filename {file} does not match expected pattern.")

                    # Sort by timestamp and store the sorted data
                    temp_storage.sort()
                    for timestamp, avg_prec_1, avg_prec_10 in temp_storage:
                        results[datasetName][oracle][algo][foldID]['timestamps'].append(timestamp)
                        results[datasetName][oracle][algo][foldID]['avg_precision_1'].append(avg_prec_1)
                        results[datasetName][oracle][algo][foldID]['avg_precision_10'].append(avg_prec_10)
                    
                    # Apply cumulative sum if requested
                    if cumulative:
                        result = results[datasetName][oracle][algo][foldID]
                        result['avg_precision_1'] = list(np.cumsum(result['avg_precision_1']))
                        result['avg_precision_10'] = list(np.cumsum(result['avg_precision_10']))

    return results

def collect_all_algorithms(results):
    """
    Collects all unique algorithm names from the results.
    """
    algorithms_set = set()
    for datasetName, oracles in results.items():
        for oracle, algorithms in oracles.items():
                for algo in algorithms.keys():
                    algorithms_set.add(algo)
    return sorted(algorithms_set)

def create_algorithm_color_mapping(algorithms):
    """
    Creates a consistent color mapping for algorithms using a high-contrast color palette.
    """
    num_algorithms = len(algorithms)
    # Use a contrasting palette like "Set1" or "bright"
    color_palette = sns.color_palette("bright", num_algorithms)  # You can also try "bright" or other high-contrast palettes
    algorithm_color_mapping = dict(zip(algorithms, color_palette))
    return algorithm_color_mapping


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Increase the size of ticks and labels
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14
})

# Custom mapping for labels
legend_label_mapping = {
    "TopTwoRules-0.0": r"$\textsc{Hss}$",
    "ScoreDifference-0.0": r"$\textsc{GUS-affine}$",
    "BradleyTerry-0.0": r"$\textsc{GUS-logistic}$",
    "Thurstone-0.0": r"$\textsc{GUS-cdf}$",
    "ChoquetRank-0.0": r"$\textsc{ChoquetRank}$"
}


# Function to save the legend as a horizontal legend
def save_horizontal_legend(lines, labels, filename="legend.pdf", title="Legend", title_fontsize=16):
    """
    Saves the legend of the provided Axes object as a horizontal PDF figure with minimal vertical space.
    """
    # Replace the labels using the mapping
    labels = [legend_label_mapping.get(label, label) for label in labels]

    fig_legend = plt.figure(figsize=(len(labels), 0.5))  # width depends on labels, height is minimal
    fig_legend.legend(lines, labels, loc='center', ncol=len(labels), title=title, title_fontsize=title_fontsize)

    fig_legend.gca().set_axis_off()
    fig_legend.savefig(filename, bbox_inches='tight')
    plt.close(fig_legend)

def plot_metrics(results, algorithm_color_mapping, cumulative):
    """
    Generates and saves plots for average precision at 1% and 10% from the results data,
    plotting metrics for each algorithm on the same graph with different colors and legend.
    
    If cumulative is True, plots cumulative metrics.
    """
    print("Plotting metrics...")
    for datasetName, oracles in results.items():
        for oracle, algorithms in oracles.items():
            print(f"Creating plots for Dataset: {datasetName}, Oracle: {oracle}")
            data_precision_1, data_precision_10 = [], []

            for algo, folds_data in algorithms.items():
                precision_1_data_algo, precision_10_data_algo = defaultdict(list), defaultdict(list)

                # Collect data across folds for each iteration
                for foldID, metrics in folds_data.items():
                    for i, (prec_1, prec_10) in enumerate(zip(metrics['avg_precision_1'], metrics['avg_precision_10'])):
                        precision_1_data_algo[i].append(prec_1)
                        precision_10_data_algo[i].append(prec_10)

                # Accumulate data from all iterations for this algorithm
                for iteration, prec_1_values in precision_1_data_algo.items():
                    for prec_1 in prec_1_values:
                        data_precision_1.append({
                            'Algorithm': algo,
                            'Iteration': iteration + 1,
                            f'{"Cumulative " if cumulative else ""}Average Precision 1%': prec_1
                        })

                for iteration, prec_10_values in precision_10_data_algo.items():
                    for prec_10 in prec_10_values:
                        data_precision_10.append({
                            'Algorithm': algo,
                            'Iteration': iteration + 1,
                            f'{"Cumulative " if cumulative else ""}Average Precision 10%': prec_10
                        })

            # Create DataFrames for plotting
            df_precision_1 = pd.DataFrame(data_precision_1)
            df_precision_10 = pd.DataFrame(data_precision_10)

            # Create a figure with two subplots side by side
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            # Plot Average Precision at 1%
            sns.lineplot(
                ax=axes[0],
                x='Iteration',
                y=f'{"Cumulative " if cumulative else ""}Average Precision 1%',
                hue='Algorithm',
                data=df_precision_1,
                palette=algorithm_color_mapping,
                marker='o',
                legend=False
            )
            axes[0].set_title(f"{datasetName} - {oracle} - {'Cumulative ' if cumulative else ''}Average Precision at 1\%")
            axes[0].set_xlabel('Iteration Number')
            axes[0].set_ylabel(f'{"Cumulative " if cumulative else ""}Average Precision at 1\%')

            # Plot Average Precision at 10%
            sns.lineplot(
                ax=axes[1],
                x='Iteration',
                y=f'{"Cumulative " if cumulative else ""}Average Precision 10%',
                hue='Algorithm',
                data=df_precision_10,
                palette=algorithm_color_mapping,
                marker='o',
            )
            
            handles, labels = axes[1].get_legend_handles_labels()
            axes[1].legend_.remove() 

            axes[1].set_title(f"{datasetName} - {oracle} - {'Cumulative ' if cumulative else ''}Average Precision at 10\%")
            axes[1].set_xlabel('Iteration Number')
            axes[1].set_ylabel(f'{"Cumulative " if cumulative else ""}Average Precision at 10\%')

            # Adjust layout and save the plot
            plt.tight_layout()
            os.makedirs("results/active_learning/output_plots/", exist_ok=True)
            output_filename = "results/active_learning/output_plots/" + f"{datasetName}_{oracle}_{'Cumulative_' if cumulative else ''}AveragePrecision_1_10_percent.pdf"
            plt.savefig(output_filename, format='pdf')

            # Save the horizontal legend with custom labels
            save_horizontal_legend(handles, labels, filename=f"{datasetName}_legend.pdf", title="Algorithms Legend")
            
            plt.close()
            print(f"Plot and legend saved to {output_filename} and {datasetName}_legend.pdf")

def main():
    parser = argparse.ArgumentParser(description='Process CSV files to compute metrics and generate plots.')
    parser.add_argument('directory', type=str, help='Directory containing CSV files.')
    parser.add_argument('--cumulative', action='store_true', help='Plot cumulative metrics if this flag is set.')
    args = parser.parse_args()

    data_dir = args.directory
    cumulative = args.cumulative

    if not os.path.isdir(data_dir):
        print(f"The directory {data_dir} does not exist or is not a directory.")
        sys.exit(1)

    grouped_files = group_files(data_dir)
    if not grouped_files:
        print("No files were grouped. Please check if the directory contains CSV files with the correct naming pattern.")
        sys.exit(1)

    results = process_files(grouped_files, cumulative)
    if not results:
        print("No results were generated. Please check if the CSV files contain the required data.")
        sys.exit(1)

    # Collect all unique algorithms
    all_algorithms = collect_all_algorithms(results)
    print(f"All algorithms: {all_algorithms}")

    # Create a consistent color mapping for algorithms
    algorithm_color_mapping = create_algorithm_color_mapping(all_algorithms)
    print(f"Algorithm color mapping: {algorithm_color_mapping}")

    plot_metrics(results, algorithm_color_mapping, cumulative)

if __name__ == "__main__":
    main()