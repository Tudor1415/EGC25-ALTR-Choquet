import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_results(results, output_base, measure, top_k):
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
    ax.set_title(f'Dataset Evaluation Results {measure} for Top {top_k} Samples')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"{measure}.png"))
    plt.close()
    print(f"Plot saved for {measure}.")
    
def generate_latex_table(results, output_base, measure, top_k):
    """
    Generate a LaTeX table from the results of dataset evaluations.

    Parameters:
    results (dict): A dictionary where the keys are dataset names and the values are
                    tuples of (best_score_tree, best_score_sample).
    output_base (str): The directory where the LaTeX table file will be saved.
    measure (str): The performance measure used (e.g., accuracy, precision).
    top_k (int): The top K samples considered.
    """
    # Import necessary modules
    import os

    # Initialize the LaTeX table string
    latex_table = ''
    latex_table += '\\begin{table}[ht]\n'
    latex_table += '\\centering\n'
    latex_table += f'\\caption{{Dataset Evaluation Results {measure} for Top {top_k} Samples}}\n'

    # Extract dataset names
    dataset_names = list(results.keys())

    # Build the tabular format string
    num_columns = len(dataset_names) + 1  # +1 for the 'Method' column
    alignment = 'l' + 'c' * len(dataset_names)
    latex_table += f'\\begin{{tabular}}{{{alignment}}}\n'
    latex_table += '\\toprule\n'

    # Write the header row
    header_row = 'Method & ' + ' & '.join(dataset_names) + ' \\\\\n'
    latex_table += header_row
    latex_table += '\\midrule\n'

    # Initialize method rows
    tree_row = 'Tree & '
    sample_row = 'Sample & '

    # Iterate over each dataset and format the scores
    for dataset in dataset_names:
        tree_score, sample_score = results[dataset]
        # Format the scores to four decimal places
        tree_score_formatted = f'{tree_score:.4f}'
        sample_score_formatted = f'{sample_score:.4f}'

        # Bold the best score for each dataset
        if tree_score > sample_score:
            tree_score_str = f'\\textbf{{{tree_score_formatted}}}'
            sample_score_str = f'{sample_score_formatted}'
        else:
            tree_score_str = f'{tree_score_formatted}'
            sample_score_str = f'\\textbf{{{sample_score_formatted}}}'

        tree_row += f'{tree_score_str} & '
        sample_row += f'{sample_score_str} & '

    # Remove trailing ampersands and add line breaks
    tree_row = tree_row.rstrip(' & ') + ' \\\\\n'
    sample_row = sample_row.rstrip(' & ') + ' \\\\\n'

    # Add the method rows to the LaTeX table
    latex_table += tree_row
    latex_table += sample_row

    # Close the table environment
    latex_table += '\\bottomrule\n'
    latex_table += '\\end{tabular}\n'
    latex_table += '\\end{table}\n'

    # Save the LaTeX table to a file
    output_file = os.path.join(output_base, f"{measure}_table.tex")
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table saved for {measure} at {output_file}.")

