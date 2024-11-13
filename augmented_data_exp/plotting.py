# plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(results, output_base):
    # Get the list of models from the results dictionary
    models_set = set()
    for dataset_name in results:
        acc_orig = results[dataset_name]['Original']
        models_set.update(acc_orig.keys())
        break  # Assuming models are consistent across datasets

    models = list(models_set)

    for model_name in models:
        plot_data = []

        for dataset_name in results.keys():
            acc_orig = results[dataset_name]['Original']
            acc_aug = results[dataset_name]['Augmented']

            # Get accuracies for the current model
            accs_orig_model = acc_orig.get(model_name, [])
            accs_aug_model = acc_aug.get(model_name, [])

            # Skip datasets without valid results
            if not accs_orig_model and not accs_aug_model:
                print(f"No valid results for dataset {dataset_name} with model {model_name}, skipping.")
                continue

            # Add original accuracies to the plot data
            if accs_orig_model:
                for acc in accs_orig_model:
                    plot_data.append({'Dataset_Type': f'{dataset_name}_Original', 'Accuracy': acc})

            # Add augmented accuracies to the plot data
            if accs_aug_model:
                for acc in accs_aug_model:
                    plot_data.append({'Dataset_Type': f'{dataset_name}_Augmented', 'Accuracy': acc})

        if not plot_data:
            print(f"No data to plot for model {model_name}.")
            continue

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
        plt.title(f'Accuracy Comparison Across Datasets for {model_name}')
        plt.ylabel('Accuracy')
        plt.xlabel('Dataset_Type')

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_base, f"accuracy_boxplot_{model_name}.png"))
        plt.close()
        print(f"Plot saved for model {model_name}.")