import os
import shutil
import json

def generate_uncertainty_sampling_config(base_config, output_path, change_period, certainty_type, start_uncertainty, max_iterations):
    """Generate Uncertainty Sampling Config with specific settings."""
    config = base_config.copy()
    config["changePeriod"] = change_period
    config["certaintyType"] = certainty_type
    config["startUncertainty"] = start_uncertainty
    config["maximumIterations"] = max_iterations
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated UncertaintySamplingConfig with changePeriod={change_period}, certaintyType={certainty_type}, startUncertainty={start_uncertainty}, maximumIterations={max_iterations} at {output_path}")

def generate_experiment_config(base_config, output_path, uncertainty_config_path, name):
    """Generate Experiment Config with updated uncertainty sampling config path."""
    config = base_config.copy()
    config["experimentName"] = name
    config["querySelectionConfigPaths"] = [
        uncertainty_config_path
    ]
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated ExperimentConfig referencing {uncertainty_config_path} at {output_path}")

def delete_folders(base_dir):
    """Deletes 'query_selection' and 'experiments' folders from the given base directory."""
    folders_to_delete = ["query_selection", "experiments"]
    for folder_name in folders_to_delete:
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            except Exception as e:
                print(f"Failed to delete folder: {folder_path}. Error: {e}")
        else:
            print(f"Folder does not exist: {folder_path}")

def main():
    # Base configuration template for uncertainty sampling
    uncertainty_base_config = {
        "noise": 0.0,
        "maximumIterations": 250,  # This will be overridden
        "certaintyType": "Sensitivity",  # This will be overridden
        "normalizationMethod": "MIN_MAX_SCALING",
        "nbLearningIterations": 100,
        "changePeriod": 10,  # This will be overridden
        "startUncertainty": False  # This will be overridden
    }

    # Base configuration template for experiments
    experiment_base_config = {
        "experimentName": "AblationStudyChangePeriod",
        "dataDirectory": "data/folds/",
        "loggingPath": "results/",
        "datasetNames": ["adult", "bank", "connect"],
        "measureNames": ["yuleQ", "cosine", "kruskal", "pavillon", "certainty"],
        "oracles": ["ChiSquaredOracle", "InformationGainOracle"],
        "learningToRankAlgorithms": ["KappalabIterative"],
        "querySelectionAlgorithms": ["UncertaintySampling"],
        "querySelectionConfigPaths": [],
        "nbLearningIterations": 100,
        "nbParallelThreads": 10,
        "testSetSize": 1000,
        "maxAntSize": 10,
        "logToFile": True
    }

    base_dir = "experimental_configs/active_learning"
    query_selection_dir = os.path.join(base_dir, "query_selection")
    experiments_dir = os.path.join(base_dir, "experiments")
    delete_folders(base_dir)
    
    # Ensure directories exist
    os.makedirs(query_selection_dir, exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)

    certainty_types = ["Sensitivity", "BradleyTerry"]
    change_periods = [(10, False, 250), (50, False, 250), (100, True, 250), (10, False, 500), (50, False, 500), (100, True, 500)]

    # Generate configs for each combination of certainty type, change period, and max iterations
    for certainty_type in certainty_types:
        for change_period, start_uncertainty, max_iterations in change_periods:
            uncertainty_config_path = os.path.join(query_selection_dir, f"{certainty_type}_uncertainty_{change_period}_{max_iterations}.json")
            experiment_config_path = os.path.join(experiments_dir, f"{certainty_type}_exp_{change_period}_{max_iterations}.json")

            # Generate Uncertainty Sampling Config
            generate_uncertainty_sampling_config(
                uncertainty_base_config,
                uncertainty_config_path,
                change_period,
                certainty_type,
                start_uncertainty,
                max_iterations
            )

            # Generate Experiment Config
            generate_experiment_config(
                experiment_base_config,
                experiment_config_path,
                uncertainty_config_path,
                f"{certainty_type}_ChangePeriod_{change_period}_MaxIter_{max_iterations}"
            )

if __name__ == "__main__":
    main()
