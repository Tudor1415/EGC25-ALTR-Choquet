import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_results(results, output_base, measure):
    """
    Visualize the results of dataset evaluations as paired bar plots.
    
    Parameters:
    results (dict): A dictionary where the keys are dataset names and the values are
                   tuples of (best_score_tree, best_score_sample).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    dataset_names = list(results.keys())
    tree_scores = [result[0] for result in results.values()]
    sample_scores = [result[1] for result in results.values()]

    x = np.arange(len(dataset_names))
    width = 0.35

    ax.bar(x - width/2, tree_scores, width, label='Tree')
    ax.bar(x + width/2, sample_scores, width, label='SiMAS')

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=90)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score')
    ax.set_title(f'Dataset Evaluation Results {measure}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"{measure}.png"))
    plt.close()
    print(f"Plot saved for {measure}.")