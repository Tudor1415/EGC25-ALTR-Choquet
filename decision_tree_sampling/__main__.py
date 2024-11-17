# main.py
import re
import os
import argparse
import numpy as np
import pandas as pd
from plotting import plot_results, generate_latex_table
from java_runner import run_extract_sample
from maven_utils import get_maven_classpath
from model_evaluation import evaluate_datasets, plot_mean_distance_cdfs

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run ExtractSample Java program on datasets.')
    parser.add_argument('--nb_samples', type=int, default=1_000_000, help='Number of samples to generate')
    parser.add_argument('--top_k', type=int, default=90, help='Top samples to compute the metrics on')
    args = parser.parse_args()

    nb_samples = args.nb_samples
    top_k = args.top_k

    # Paths relative to the scripts folder
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(exp_dir, '..'))
    dat_files_folder = os.path.join(project_root, 'data', 'dat-files') + "/"
    target_dir = os.path.join(project_root, 'target')
    class_output_dir = os.path.join(target_dir, 'classes')
    output_base = os.path.join(project_root, 'results', 'decision_tree_sampling')

    # Timeout in minutes
    timeout_minutes = 30

    # Measure names and weights (can be adjusted as needed)
    measure_names = ['phi', 'IG']
    weights = '1'

    # Step 1: Get the classpath of the project including dependencies
    maven_classpath = get_maven_classpath(project_root)
    
    # Combine the Maven classpath with the compiled classes in target/classes
    execution_classpath = os.pathsep.join([class_output_dir, maven_classpath])

    # Step 2: Iterate over all dataset files in the dataset folder and run Java program
    for measure in measure_names:
        # for dataset_file in os.listdir(dat_files_folder):
        #     if os.path.isfile(os.path.join(dat_files_folder, dataset_file)):
        #         dataset_name = dataset_file
        #         # Construct the output directory
        #         dataset_name_without_extension = os.path.splitext(dataset_name)[0]
        #         output_directory = os.path.join(output_base, dataset_name_without_extension, 'samples', measure) + "/"

        #         # Ensure the output directory exists
        #         os.makedirs(output_directory, exist_ok=True)
        #         # Run the Java program
        #         run_extract_sample(
        #             dataset_name,
        #             dat_files_folder,
        #             output_directory,
        #             nb_samples,
        #             timeout_minutes,
        #             measure,
        #             weights,
        #             execution_classpath,
        #             project_root
        #         )

        # Evaluate datasets and collect data
        results, tree_data_per_dataset, sample_data_per_dataset = evaluate_datasets(dat_files_folder, output_base, measure, top_k)

        plot_mean_distance_cdfs(tree_data_per_dataset, sample_data_per_dataset, output_base, measure)
        generate_latex_table(results, output_base, measure, top_k)
        plot_results(results, output_base, measure, top_k)

    print("Processing complete.")

if __name__ == '__main__':
    main()