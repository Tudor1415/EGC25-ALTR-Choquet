import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Function to group files in a nested dictionary structure
def group_files_in_nested_dict(base_folder):
    grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    filename_without_extension = filename[:-4]
                    parts = filename_without_extension.split('_')
                    timestamp = parts[-1]
                    outRankingCertainty = parts[-2]
                    numIterations = parts[-3]
                    Oracle = parts[-4]
                    foldIdx = parts[-5]
                    datasetName = '_'.join(parts[:-5])
                    
                    grouped_files[datasetName][outRankingCertainty][numIterations].append(os.path.join(folder_path, filename))
    
    return grouped_files

# Function to compute mean and variance of the best scores
def compute_mean_variance_for_best_scores(grouped_files):
    results = defaultdict(lambda: defaultdict(lambda: {}))
    
    for datasetName, outRankingDict in grouped_files.items():
        for outRankingCertainty, numIterationDict in outRankingDict.items():
            for numIterations, files in numIterationDict.items():
                best_scores = []
                
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        if df.empty:
                            best_score = 0
                        else:
                            best_score = df['scoreApprox'].max()
                        best_scores.append(best_score)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue

                if best_scores:
                    mean_score = np.mean(best_scores)
                    variance_score = np.var(best_scores)
                    min_score = np.min(best_scores)
                    max_score = np.max(best_scores)

                    results[datasetName][outRankingCertainty][numIterations] = {
                        'mean': mean_score,
                        'variance': variance_score,
                        'min': np.abs(mean_score - min_score),
                        "max": np.abs(mean_score - max_score)
                    }
    
    return results

# Enable LaTeX-style text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Mapping of the old labels to the new legend labels
legend_label_mapping = {
    "ScoreDifferenceOutRanking": r"$\textsc{SiMAS-affine}$",
    "ThurstoneOutRanking": r"$\textsc{Simas-cdf}$",
    "BradleyTerryOutRanking": r"$\textsc{Simas-logistic}$",
    "UnrestrictedSampling": r"$\textsc{UnrestrictedSampling}$"
}

# Updated save_horizontal_legend function with LaTeX support
def save_horizontal_legend(ax, filename="legend.pdf", title="Legend", title_fontsize=16):
    """
    Saves the legend of the provided Axes object as a horizontal PDF figure with minimal vertical space.
    """
    lines, labels = ax.get_legend_handles_labels()

    # Replace the labels using the mapping
    labels = [legend_label_mapping.get(label, label) for label in labels]

    fig_legend = plt.figure(figsize=(len(labels), 0.5))  # width depends on labels, height is minimal
    fig_legend.legend(lines, labels, loc='center', ncol=len(labels), title=title, title_fontsize=title_fontsize)

    fig_legend.gca().set_axis_off()
    fig_legend.savefig(filename, bbox_inches='tight')
    plt.close(fig_legend)


# Updated plotting function to compute mean cumulative distribution and save the side-by-side plots to a file
def plot_results_and_save(mean_variance_results, grouped_files, output_folder):
    for datasetName, outRankingDict in mean_variance_results.items():
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))  # Create three subplots side by side

        # First plot (ax1): Maximum score vs iterations with error bars
        for outRankingCertainty, numIterationDict in outRankingDict.items():
            num_iterations = []
            means = []
            errors = []

            for numIterations, stats in sorted(numIterationDict.items(), key=lambda x: int(x[0])):
                num_iterations.append(int(numIterations))
                means.append(stats['mean'])
                errors.append([stats['min'], stats['max']])
                
            ax1.errorbar(num_iterations, means, yerr=np.array(errors).T, capsize=5, marker='o')

        # Set log scale for the x-axis
        ax1.set_xscale('log')
        ax1.set_xlabel('Value of k', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Maximum Score', fontsize=18, fontweight='bold')
        ax1.set_title(f'Maximum Score ({datasetName})', fontdict={'fontsize': 20, 'fontweight': 'bold'})

        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # Second plot (ax2): Average number of samples vs iterations with error bars
        for outRankingCertainty, numIterationDict in outRankingDict.items():
            num_iterations = []
            num_samples_means = []
            num_samples_errors = []
            
            for numIterations, files in sorted(numIterationDict.items(), key=lambda x: int(x[0])):
                num_iterations.append(int(numIterations))
                num_samples = []
                
                for file in grouped_files[datasetName][outRankingCertainty][numIterations]:
                    try:
                        df = pd.read_csv(file)
                        if df.empty:
                            num_samples.append(1)  
                        else:
                            num_samples.append(len(df)) 
                            
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue

                if num_samples:
                    num_samples_means.append(np.mean(num_samples))
                    num_samples_errors.append([np.abs(np.min(num_samples) - np.mean(num_samples)), np.abs(np.max(num_samples) - np.mean(num_samples))])

            # Plot with error bars for number of samples on ax2
            ax2.errorbar(num_iterations, num_samples_means, yerr=np.array(num_samples_errors).T, label=outRankingCertainty, capsize=5, marker='o', linestyle='--')

        # Set log scale for the x-axis
        ax2.set_xscale('log')
        ax2.set_xlabel('Value of k', fontsize=18, fontweight='bold')
        ax2.set_yscale('log')
        ax2.set_ylabel('Nb. Samples', fontsize=18, fontweight='bold')
        ax2.set_title(f'Distinct samples ({datasetName})', fontdict={'fontsize': 20, 'fontweight': 'bold'})

        # Set the tick parameters to increase the size of the ticks
        ax2.tick_params(axis='both', which='major', labelsize=16)

        # Set legend title styling
        save_horizontal_legend(ax2, output_folder + f"/legend_{datasetName}.pdf", title="")

        # Third plot (ax3): Mean cumulative distribution at 1000000 iterations with error bars
        iteration_target = '100000'
        for outRankingCertainty, numIterationDict in outRankingDict.items():
            if iteration_target in numIterationDict:
                cumulative_values = []
                files = grouped_files[datasetName][outRankingCertainty][iteration_target]
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        if df.empty:
                            cumulative_values.extend([0])
                        else:
                            cumulative_values.extend(df['scoreApprox'])
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
                
                if np.all(np.array(cumulative_values) == 0):
                    sorted_values = [0, 3] 
                    cumulative_dist = [0, 1]  
                else:
                    sorted_values = np.sort(cumulative_values)
                    cumulative_dist = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

                # Plot the cumulative distribution as a step plot
                ax3.step(sorted_values, cumulative_dist, label=outRankingCertainty, linestyle='-', marker='')

        ax3.set_title(f'Score c.d.f. ({datasetName})', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        ax3.set_xlabel('Score \(\\tau\)', fontsize=18, fontweight='bold')
        ax3.set_ylabel('C.d.f', fontsize=18, fontweight='bold')

        # Set the tick parameters to increase the size of the ticks
        ax3.tick_params(axis='both', which='major', labelsize=16)
        
        # Adjust layout and save the plot to file
        plt.tight_layout()
        output_file = os.path.join(output_folder, f'{datasetName}_side_by_side_plots.pdf')
        plt.savefig(output_file, format='pdf')
        plt.close(fig)

# Replace with your actual path to the data folder
base_folder = "results/sampling_experiment/samples/ratio_1to1/"

# Group files
nested_grouped_files = group_files_in_nested_dict(base_folder)

# Compute mean and variance for each numIterations group
mean_variance_results = compute_mean_variance_for_best_scores(nested_grouped_files)

# Define the output folder where the plots will be saved
output_folder = "results/sampling_experiment/output_plots/ratio_1to1"
os.makedirs(output_folder, exist_ok=True)

# Call the updated plot function to generate and save the side-by-side plots
plot_results_and_save(mean_variance_results, nested_grouped_files, output_folder)
