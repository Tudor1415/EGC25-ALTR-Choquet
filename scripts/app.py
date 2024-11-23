import os
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from plot_active_learning import group_files

# Set global plot style
sns.set(style="whitegrid")

# Set the directory for results
RESULTS_DIR = "results"

def initialize_results_structure():
    """
    Creates an explicitly initialized nested dictionary for storing metrics.
    """
    return {
        "avg_precision_1": [],
        "avg_precision_10": [],
        "timestamps": []
    }


def process_files(grouped_files, cumulative=False):
    """
    Processes files and computes metrics with explicitly initialized dictionaries.
    """
    results = {}

    top_percentage_1 = 0.01
    top_percentage_10 = 0.10

    for dataset_name, oracles in grouped_files.items():
        if dataset_name not in results:
            results[dataset_name] = {}

        for oracle, algorithms in oracles.items():
            if oracle not in results[dataset_name]:
                results[dataset_name][oracle] = {}

            for algorithm, folds in algorithms.items():
                if algorithm not in results[dataset_name][oracle]:
                    results[dataset_name][oracle][algorithm] = {}

                for fold_id, file_list in folds.items():
                    if fold_id not in results[dataset_name][oracle][algorithm]:
                        results[dataset_name][oracle][algorithm][fold_id] = initialize_results_structure()

                    temp_storage = []

                    for file_path in file_list:
                        try:
                            df = pd.read_csv(file_path)

                            if "scoreApprox" not in df.columns or "scoreOracle" not in df.columns:
                                continue

                            # Compute rankings
                            df["rankApprox"] = df["scoreApprox"].rank(ascending=False, method="first")
                            df["rankOracle"] = df["scoreOracle"].rank(ascending=False, method="first")

                            k_1 = max(1, int(len(df) * top_percentage_1))
                            k_10 = max(1, int(len(df) * top_percentage_10))

                            avg_precision_1 = df.nsmallest(k_1, "rankApprox")["rankOracle"].le(k_1).mean()
                            avg_precision_10 = df.nsmallest(k_10, "rankApprox")["rankOracle"].le(k_10).mean()

                            timestamp = int(os.path.basename(file_path).split("_")[-1].split(".")[0])
                            temp_storage.append((timestamp, avg_precision_1, avg_precision_10))
                        except Exception as e:
                            st.warning(f"Error processing file {file_path}: {e}")
                            continue

                    temp_storage.sort()

                    for timestamp, avg_prec_1, avg_prec_10 in temp_storage:
                        results[dataset_name][oracle][algorithm][fold_id]["timestamps"].append(timestamp)
                        results[dataset_name][oracle][algorithm][fold_id]["avg_precision_1"].append(avg_prec_1)
                        results[dataset_name][oracle][algorithm][fold_id]["avg_precision_10"].append(avg_prec_10)

                    if cumulative:
                        result = results[dataset_name][oracle][algorithm][fold_id]
                        result["avg_precision_1"] = list(pd.Series(result["avg_precision_1"]).cumsum())
                        result["avg_precision_10"] = list(pd.Series(result["avg_precision_10"]).cumsum())

    return results


@st.cache_data(show_spinner=True)
def precompute_metrics(results_dir):
    """
    Precomputes all metrics for valid periods in the results directory.
    Returns a dictionary of precomputed results and a sorted list of valid periods.
    """
    precomputed_results = {}
    valid_periods = []

    for folder_name in os.listdir(results_dir):
        if folder_name.startswith("Exp_ChangePeriod_"):
            try:
                period = int(folder_name.split("_")[-1])
                valid_periods.append(period)

                folder_path = os.path.join(results_dir, folder_name, "samples")
                grouped_files = group_files(folder_path)

                results = process_files(grouped_files)
                precomputed_results[period] = results

            except ValueError:
                st.warning(f"Invalid period format in directory: {folder_name}")
            except Exception as e:
                st.error(f"Error processing directory {folder_name}: {e}")

    valid_periods = sorted(valid_periods)
    return precomputed_results, valid_periods


# Precompute metrics during app initialization
st.title("Precomputing Metrics")
precomputed_results, valid_periods = precompute_metrics(RESULTS_DIR)

# Show error if no valid periods are found
if not valid_periods:
    st.error("No valid period directories found in the results folder.")
    st.stop()

# Sidebar: Select Period
selected_period = st.sidebar.slider(
    "Select Period",
    min_value=min(valid_periods),
    max_value=max(valid_periods),
    step=1,
    value=min(valid_periods),
)

# Retrieve precomputed metrics for the selected period
results = precomputed_results.get(selected_period, {})
if not results:
    st.error(f"No data available for period {selected_period}.")
    st.stop()

# Prepare data for plotting
st.title(f"Uncertainty Plots for Period {selected_period}")
algorithms = {algo for _, oracles in results.items() for _, algos in oracles.items() for algo in algos.keys()}
algorithm_color_mapping = {algo: sns.color_palette("bright", len(algorithms))[i] for i, algo in enumerate(algorithms)}

data_precision_1 = []
data_precision_10 = []

for dataset_name, oracles in results.items():
    for oracle, algorithms in oracles.items():
        for algorithm, folds_data in algorithms.items():
            for fold_id, metrics in folds_data.items():
                for i, (prec_1, prec_10) in enumerate(zip(metrics["avg_precision_1"], metrics["avg_precision_10"])):
                    data_precision_1.append({
                        "Algorithm": algorithm,
                        "Iteration": i + 1,
                        "Average Precision 1%": prec_1
                    })
                    data_precision_10.append({
                        "Algorithm": algorithm,
                        "Iteration": i + 1,
                        "Average Precision 10%": prec_10
                    })

df_precision_1 = pd.DataFrame(data_precision_1)
df_precision_10 = pd.DataFrame(data_precision_10)

if df_precision_1.empty or df_precision_10.empty:
    st.error(f"No data available for period {selected_period}.")
    st.stop()

# Left chart: Average Precision at 10%
st.subheader("Average Precision at 10%")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.lineplot(
    x="Iteration",
    y="Average Precision 10%",
    hue="Algorithm",
    data=df_precision_10,
    palette=algorithm_color_mapping,
    marker="o",
    ax=ax1
)
ax1.set_title(f"Average Precision at 10% for Period {selected_period}")
st.pyplot(fig1)

# Right chart: Average Precision at 1%
st.subheader("Average Precision at 1%")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.lineplot(
    x="Iteration",
    y="Average Precision 1%",
    hue="Algorithm",
    data=df_precision_1,
    palette=algorithm_color_mapping,
    marker="o",
    ax=ax2
)
ax2.set_title(f"Average Precision at 1% for Period {selected_period}")
st.pyplot(fig2)
