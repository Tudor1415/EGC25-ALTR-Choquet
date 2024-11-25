#!/usr/bin/env python3

import os
import re
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_filename(filename):
    """
    Parses the filename to extract datasetName, foldID, LearningAlgorithm, Oracle, and timestamp.
    Expected filename format: datasetName_foldID_LearningAlgorithm_Oracle_timestamp.csv
    """
    logger.debug(f"Parsing filename: {filename}")
    basename = os.path.basename(filename)
    pattern = r'^(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv$'
    match = re.match(pattern, basename)
    if match:
        datasetName, foldID, LearningAlgorithm, Oracle, timestamp = match.groups()
        logger.debug(f"Parsed: {datasetName}, {foldID}, {LearningAlgorithm}, {Oracle}, {timestamp}")
        return datasetName, foldID, LearningAlgorithm, Oracle, timestamp
    else:
        logger.warning(f"Filename {filename} does not match expected pattern.")
        return None


def group_files(directory):
    """
    Groups files by datasetName, LearningAlgorithm, Oracle, and foldID in a nested dictionary.
    """
    logger.info(f"Grouping files in directory: {directory}")
    files_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            parsed = parse_filename(filename)
            if parsed:
                datasetName, foldID, LearningAlgorithm, Oracle, timestamp = parsed
                file_path = os.path.join(directory, filename)
                files_dict[datasetName][Oracle][LearningAlgorithm][foldID].append(file_path)
            else:
                logger.warning(f"Skipping file: {filename}")

    logger.info(f"Grouped files: {len(files_dict)} datasets found.")
    return files_dict


def compute_rankings(df):
    """
    Computes rankings based on scoreApprox and scoreOracle.
    """
    logger.debug("Computing rankings.")
    df = df.copy()
    df['rankApprox'] = df['scoreApprox'].rank(ascending=False, method='first')
    df['rankOracle'] = df['scoreOracle'].rank(ascending=False, method='first')
    return df


def compute_average_precision_at_k(df, k):
    """
    Computes Average Precision at top k entries.
    """
    logger.debug(f"Computing average precision at k={k}.")
    df = df.nsmallest(k, 'rankApprox')
    relevant = df['rankOracle'] <= k
    precision_at_k = relevant.sum() / k
    logger.debug(f"Average precision at k={k}: {precision_at_k}")
    return precision_at_k


def process_files(grouped_files, cumulative):
    """
    Processes each group of files and computes metrics.
    """
    logger.info("Processing files.")
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'avg_precision_1': [], 'avg_precision_10': [], 'timestamps': []
    }))))

    top_percentage_1 = 0.01
    top_percentage_10 = 0.10

    for datasetName, oracles in grouped_files.items():
        logger.info(f"Processing dataset: {datasetName}")
        for oracle, algos in oracles.items():
            logger.info(f"Processing oracle: {oracle}")
            for algo, folds in algos.items():
                logger.info(f"Processing algorithm: {algo}")
                for foldID, files in folds.items():
                    num_files = len(files)
                    logger.info(f"Processing fold: {foldID}, Number of files: {num_files}")

                    temp_storage = []

                    for file in files:
                        logger.debug(f"Reading file: {file}")
                        try:
                            df = pd.read_csv(file)
                            if 'scoreApprox' not in df.columns or 'scoreOracle' not in df.columns:
                                logger.warning(f"Missing required columns in {file}. Skipping.")
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
                        except Exception as e:
                            logger.error(f"Error processing file {file}: {e}")

                    temp_storage.sort()
                    for timestamp, avg_prec_1, avg_prec_10 in temp_storage:
                        results[datasetName][oracle][algo][foldID]['timestamps'].append(timestamp)
                        results[datasetName][oracle][algo][foldID]['avg_precision_1'].append(avg_prec_1)
                        results[datasetName][oracle][algo][foldID]['avg_precision_10'].append(avg_prec_10)

                    if cumulative:
                        result = results[datasetName][oracle][algo][foldID]
                        result['avg_precision_1'] = list(np.cumsum(result['avg_precision_1']))
                        result['avg_precision_10'] = list(np.cumsum(result['avg_precision_10']))

    logger.info("Finished processing files.")
    return results


def plot_metrics(results, algorithm_color_mapping, cumulative, use_latex):
    """
    Generates and saves plots for average precision.
    """
    logger.info("Plotting metrics.")
    if use_latex:
        logger.debug("Using LaTeX for rendering plots.")
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    else:
        plt.rc('text', usetex=False)

    for datasetName, oracles in results.items():
        for oracle, algorithms in oracles.items():
            for algo, folds_data in algorithms.items():
                logger.info(f"Generating plots for {datasetName} - {oracle} - {algo}")
                data_precision_1, data_precision_10 = [], []

                for foldID, metrics in folds_data.items():
                    for i, (prec_1, prec_10) in enumerate(zip(metrics['avg_precision_1'], metrics['avg_precision_10'])):
                        data_precision_1.append({
                            'Algorithm': algo,
                            'Iteration': i + 1,
                            f'{"Cumulative " if cumulative else ""}Average Precision 1%': prec_1
                        })
                        data_precision_10.append({
                            'Algorithm': algo,
                            'Iteration': i + 1,
                            f'{"Cumulative " if cumulative else ""}Average Precision 10%': prec_10
                        })

                df_precision_1 = pd.DataFrame(data_precision_1)
                df_precision_10 = pd.DataFrame(data_precision_10)

                fig, axes = plt.subplots(1, 2, figsize=(20, 10))

                sns.lineplot(
                    ax=axes[0],
                    x='Iteration',
                    y=f'{"Cumulative " if cumulative else ""}Average Precision 1%',
                    hue='Algorithm',
                    data=df_precision_1,
                    palette=algorithm_color_mapping,
                    marker='o'
                )
                axes[0].set_ylim(0, 1) 
                sns.lineplot(
                    ax=axes[1],
                    x='Iteration',
                    y=f'{"Cumulative " if cumulative else ""}Average Precision 10%',
                    hue='Algorithm',
                    data=df_precision_10,
                    palette=algorithm_color_mapping,
                    marker='o'
                )
                axes[1].set_ylim(0, 1) 
                output_dir = "results/output_plots/"
                os.makedirs(output_dir, exist_ok=True)
                output_filename = os.path.join(output_dir, f"{datasetName}_{oracle}_precision.pdf")
                plt.savefig(output_filename, format='pdf')
                logger.info(f"Saved plot: {output_filename}")
                plt.close()

def plot_metrics_single_figure(results, algorithm_color_mapping, output_dir, cumulative, use_latex):
    """
    Generates and saves plots for average precision with all algorithms on the same figure for each dataset and oracle.
    """
    logger.info("Plotting metrics with all algorithms on the same figure.")
    if use_latex:
        logger.debug("Using LaTeX for rendering plots.")
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    else:
        plt.rc('text', usetex=False)

    for datasetName, oracles in results.items():
        for oracle, algorithms in oracles.items():
            data_precision_1, data_precision_10 = [], []

            for algo, folds_data in algorithms.items():
                for foldID, metrics in folds_data.items():
                    for i, (prec_1, prec_10) in enumerate(zip(metrics['avg_precision_1'], metrics['avg_precision_10'])):
                        data_precision_1.append({
                            'Algorithm': algo,
                            'Iteration': i + 1,
                            f'{"Cumulative " if cumulative else ""}Average Precision 1%': prec_1
                        })
                        data_precision_10.append({
                            'Algorithm': algo,
                            'Iteration': i + 1,
                            f'{"Cumulative " if cumulative else ""}Average Precision 10%': prec_10
                        })

            # Create dataframes
            df_precision_1 = pd.DataFrame(data_precision_1)
            df_precision_10 = pd.DataFrame(data_precision_10)

            # Create plots
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            sns.lineplot(
                ax=axes[0],
                x='Iteration',
                y=f'{"Cumulative " if cumulative else ""}Average Precision 1%',
                hue='Algorithm',
                data=df_precision_1,
                palette=algorithm_color_mapping,
                marker='o'
            )
            axes[0].set_ylim(0, 1) 
            axes[0].set_title(f"{datasetName} - {oracle} - Average Precision 1%")
            axes[0].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

            sns.lineplot(
                ax=axes[1],
                x='Iteration',
                y=f'{"Cumulative " if cumulative else ""}Average Precision 10%',
                hue='Algorithm',
                data=df_precision_10,
                palette=algorithm_color_mapping,
                marker='o'
            )
            axes[1].set_ylim(0, 1) 
            axes[1].set_title(f"{datasetName} - {oracle} - Average Precision 10%")
            axes[1].legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

            # Save plots
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"{datasetName}_{oracle}_precision_all_algorithms.pdf")
            plt.savefig(output_filename, format='pdf', bbox_inches='tight')
            logger.info(f"Saved plot: {output_filename}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process CSV files to compute metrics and generate plots.')
    parser.add_argument('directory', type=str, help='Directory containing CSV files.')
    parser.add_argument('--cumulative', action='store_true', help='Plot cumulative metrics.')
    parser.add_argument('--use_latex', action='store_true', help='Use LaTeX for rendering text in plots.')

    args = parser.parse_args()

    logger.info("Starting script.")
    grouped_files = group_files(args.directory)
    results = process_files(grouped_files, args.cumulative)

    algorithms = {algo for _, oracles in results.items() for _, algos in oracles.items() for algo in algos.keys()}
    algorithm_color_mapping = {algo: sns.color_palette("bright", len(algorithms))[i] for i, algo in enumerate(algorithms)}

    output_dir = args.directory + "output_plots/"
    plot_metrics_single_figure(results, algorithm_color_mapping, output_dir, args.cumulative, args.use_latex)
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
