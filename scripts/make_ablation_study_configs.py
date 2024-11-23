import os
import json

def generate_uncertainty_sampling_config(base_config, output_path, change_period):
    """Generate Uncertainty Sampling Config with updated changePeriod."""
    config = base_config.copy()
    config["changePeriod"] = change_period  # Update changePeriod
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Generated UncertaintySamplingConfig with changePeriod={change_period} at {output_path}")


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


def main():
    # Base configuration templates
    uncertainty_base_config = {
        "noise": 0.0,
        "maximumIterations": 500,
        "certaintyType": "ScoreDifference",
        "normalizationMethod": "MIN_MAX_SCALING",
        "nbLearningIterations": 100,
        "changePeriod": 10,  # Will be overridden in each config
        "startUncertainty": False
    }

    experiment_base_config = {
        "experimentName": "AblationStudyChangePeriod",
        "dataDirectory": "data/folds/",
        "loggingPath": "results/",
        "datasetNames": ["adult", "bank", "connect", "credit", "dota", "toms"],
        "measureNames": ["yuleQ", "cosine", "kruskal", "pavillon", "certainty"],
        "oracles": ["ChiSquaredOracle", "InformationGainOracle"],
        "learningToRankAlgorithms": ["KappalabIterative"],
        "querySelectionAlgorithms": ["UncertaintySampling"],
        "querySelectionConfigPaths": [],  # Will be updated dynamically
        "nbLearningIterations": 100,
        "nbParallelThreads": 10,
        "testSetSize": 1000,
        "maxAntSize": 10,
        "logToFile": True
    }

    base_dir = "experimental_configs/active_learning"
    query_selection_dir = os.path.join(base_dir, "query_selection")
    experiments_dir = os.path.join(base_dir, "experiments")

    # Ensure directories exist
    os.makedirs(query_selection_dir, exist_ok=True)
    os.makedirs(experiments_dir, exist_ok=True)

    # Generate 10 configs
    for i in range(6):
        change_period = i*10
        uncertainty_config_path = os.path.join(query_selection_dir, f"uncertainty_sampling_conf_{change_period}.json")
        experiment_config_path = os.path.join(experiments_dir, f"exp_{change_period}.json")

        # Generate Uncertainty Sampling Config
        generate_uncertainty_sampling_config(
            uncertainty_base_config,
            uncertainty_config_path,
            change_period
        )

        # Generate Experiment Config
        generate_experiment_config(
            experiment_base_config,
            experiment_config_path,
            uncertainty_config_path,
            f"Exp_ChangePeriod_{change_period}"
        )


if __name__ == "__main__":
    main()
