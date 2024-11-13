# main.py
import os
import argparse
from maven_utils import get_maven_classpath
from java_runner import run_extract_sample
from data_processing import process_datasets
from model_evaluation import evaluate_datasets
from plotting import plot_results
from scripts.augmented_data_experiment import dataset_folder

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run ExtractSample Java program on datasets.')
    parser.add_argument('--nb_samples', type=int, default=100_000, help='Number of samples to generate')
    args = parser.parse_args()

    nb_samples = args.nb_samples
    nb_folds = 5

    # Paths relative to the scripts folder
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(scripts_dir, '..'))
    dataset_folder = os.path.join(project_root, 'data', 'datasets')
    dat_file_folder = os.path.join(project_root, 'data', 'dat-files') + "\\"
    mapping_file_folder = os.path.join(project_root, 'data', 'mapping-files')
    target_dir = os.path.join(project_root, 'target')
    class_output_dir = os.path.join(target_dir, 'classes')
    output_base = os.path.join(project_root, 'results', 'augmented_data_exp')

    # Timeout in minutes
    timeout_minutes = 30

    # Measure names and weights (can be adjusted as needed)
    measure_names = 'phi'
    weights = '1'

    # Step 1: Get the classpath of the project including dependencies
    maven_classpath = get_maven_classpath(project_root)
    
    # Combine the Maven classpath with the compiled classes in target/classes
    execution_classpath = os.pathsep.join([class_output_dir, maven_classpath])

    # Step 2: Iterate over all dataset files in the dataset folder and run Java program
    for dataset_file in os.listdir(dat_file_folder):
        if os.path.isfile(os.path.join(dat_file_folder, dataset_file)):
            dataset_name = dataset_file
            # Construct the output directory
            dataset_name_without_extension = os.path.splitext(dataset_name)[0]
            output_directory = os.path.join(output_base, dataset_name_without_extension, 'samples')

            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)

            # Run the Java program
            run_extract_sample(
                dataset_name,
                dat_file_folder,
                output_directory,
                nb_samples,
                timeout_minutes,
                measure_names,
                weights,
                execution_classpath,
                project_root
            )

    # Process datasets to add rule columns
    process_datasets(dat_file_folder, output_base)

    # Evaluate datasets
    results = evaluate_datasets(dat_file_folder, output_base, nb_folds)

    # Plot results
    plot_results(results, output_base)

    print("Processing complete.")

if __name__ == '__main__':
    main()
