# plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(results, output_base):
    plot_data = []
    
    for dataset_name in results.keys():
        acc_orig = results[dataset_name]['Original']
        acc_aug = results[dataset_name]['Augmented']

        # Skip datasets without valid results
        if not acc_orig and not acc_aug:
            print(f"No valid results for dataset {dataset_name}, skipping.")
            continue

        # Add original accuracies to the plot data
        if acc_orig:
            for acc in acc_orig:
                plot_data.append({'Dataset_Type': f'{dataset_name}_Original', 'Accuracy': acc})

        # Add augmented accuracies to the plot data
        if acc_aug:
            for acc in acc_aug:
                plot_data.append({'Dataset_Type': f'{dataset_name}_Augmented', 'Accuracy': acc})

    # Convert the plot data to a DataFrame
    df_plot = pd.DataFrame(plot_data)

    # Prepare data for matplotlib boxplot
    data_to_plot = []
    labels = []

    for dataset_type in df_plot['Dataset_Type'].unique():
        data_to_plot.append(df_plot[df_plot['Dataset_Type'] == dataset_type]['Accuracy'].values)
        labels.append(dataset_type)

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Create the boxplot
    bp = plt.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "orange",
            "markeredgecolor": "black"
        }
    )

    # Set properties for each box
    for box in bp['boxes']:
        # Change outline color
        box.set(color='black', linewidth=1)
        # Change fill color with alpha
        box.set(facecolor=(1, 1, 1, 0.3))  # White with alpha=0.3

    # Set color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)

    # Set color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)

    # Set color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='black', linewidth=1)

    # Set style for the means
    for mean in bp['means']:
        mean.set(marker='D', markerfacecolor='orange', markeredgecolor='black')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set the title and labels
    plt.title('Accuracy Comparison Across Datasets')
    plt.ylabel('Accuracy')
    plt.xlabel('Dataset_Type')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_base, f"all_datasets_accuracy_boxplot.png"))
    plt.close()
    print("Combined plot saved for all datasets.")
