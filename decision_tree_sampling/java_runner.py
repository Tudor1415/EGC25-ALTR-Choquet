# java_runner.py
import sys
import subprocess

def run_extract_sample(dataset_name, dataset_folder, output_directory, nb_samples, timeout_minutes, measure_names, weights, execution_classpath, project_root):
    command = [
        'java',
        '-cp',
        execution_classpath,
        'tools.utils.ExtractSample',
        dataset_name,
        dataset_folder,
        output_directory,
        str(nb_samples),
        str(timeout_minutes),
        measure_names,
        weights
    ]

    print(f"Processing dataset {dataset_name} with {nb_samples} samples...")
    # Call the Java program
    try:
        subprocess.run(command, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing {dataset_name}: {e}")
    except FileNotFoundError:
        print("Java executable not found. Please ensure Java is installed and added to your system's PATH.")
        sys.exit(1)