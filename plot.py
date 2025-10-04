import pandas as pd
import matplotlib.pyplot as plt
import re
import os # os is used for checking if files exist
import seaborn as sns
import re
import glob
import numpy as np
from scipy.stats import cumfreq


# Define the experiments and their respective labels
experiments = [
        {"label": "Gamma_r", "folders": ["exp1_gamma_r=0.6", "exp1_gamma_r=1.0", "exp1_gamma_r=1.5", "exp1_gamma_r=2.0"], "gamma": None},
        {"label": "Lambda", "folders": ["exp2_lambda_true=0.5", "exp2_lambda_true=1.5", "exp2_lambda_true=5.0"], "gamma": None},
        {"label": "W (Measurement Budget)", "folders": ["exp3_W=100", "exp3_W=20", "exp3_W=500"], "gamma": None},
        {"label": "L (Sample Size)", "folders": ["exp4_L=128", "exp4_L=256", "exp4_L=64"], "gamma": None},
    ]

def compute_cvar(losses, gamma):
    """
    Compute the empirical CVaR for given losses and quantile gamma.
    :param losses: List or array of losses.
    :param gamma: Quantile for CVaR calculation (between 0 and 1).
    :return: CVaR of the losses.
    """
    sorted_losses = np.sort(losses)
    k = int(np.ceil(gamma * len(sorted_losses)))
    cvar = np.mean(sorted_losses[:k])
    return cvar

def Utility_CDFs(results_folder):
    """
    Draw 4 CDF plots for utility from 4 different experiments (gamma_r, lambda, W, L).
    Each plot will show the CDF of utility and mark CVaR and mean utility, then save them as images.
    :param results_folder: Path to the folder where results of the experiments are saved.
    """
    # Ensure the "figures" folder exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    for i, experiment in enumerate(experiments):
        # Set up a plot for each experiment
        fig, ax = plt.subplots(figsize=(7, 5))
        experiment_label = experiment["label"]
        
        # Collect utility and loss data for each folder in the experiment
        for folder in experiment["folders"]:
            folder_path = os.path.join(results_folder, folder, "checkpoints")
            file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('TRAIN')]
            
            for file_path in file_paths:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Extract utility and loss columns
                utility = df['mean_utility_mean_RL2O-CVaR']  # or another method
                loss = df['loss_train_mean']
                
                # Compute CVaR for the losses (assuming gamma=0.05 for CVaR)
                cvar = compute_cvar(loss, 0.05)
                
                # Plot the CDF of the utility (1 - loss)
                sorted_utility = np.sort(1 - loss)  # Utility is 1 - loss
                cdf = np.cumsum(np.ones_like(sorted_utility)) / len(sorted_utility)
                
                ax.plot(sorted_utility, cdf, label=f'{folder} - CDF', linewidth=2)
                ax.axvline(x=cvar, color='red', linestyle='--', label=f'CVaR (gamma=0.05)')
                ax.axvline(x=np.median(sorted_utility), color='green', linestyle=':', label='Median')
        
        # Customize the plot for this experiment
        ax.set_title(f'{experiment_label} - Utility CDFs')
        ax.set_xlabel('Utility (1 - Loss)')
        ax.set_ylabel('CDF')
        ax.legend()

        # Save the plot as an image in the "figures" folder
        plot_filename = f"figures/1_Utility_CDF_{experiment_label}.png"
        fig.savefig(plot_filename)
        plt.close(fig)  # Close the figure after saving to avoid display

    print("All plots have been saved in the 'figures' folder.")


def Mean_vs_CVaR_tradeoff_scatter(results_folder):
    """
    Draw 4 scatter plots for the tradeoff between mean utility and CVaR for different experiments (gamma_r, lambda, W, L).
    Each plot will show the relationship between mean utility and CVaR.
    :param results_folder: Path to the folder where results of the experiments are saved.
    """
    # Ensure the "figures" folder exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    for i, experiment in enumerate(experiments):
        # Set up a plot for each experiment
        fig, ax = plt.subplots(figsize=(7, 5))
        experiment_label = experiment["label"]
        
        # Collect utility and loss data for each folder in the experiment
        for folder in experiment["folders"]:
            folder_path = os.path.join(results_folder, folder, "checkpoints")
            file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('TRAIN')]
            
            for file_path in file_paths:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Extract utility and loss columns
                utility = df['mean_utility_mean_RL2O-CVaR']  # or another method
                loss = df['loss_train_mean']
                
                # Compute CVaR for the losses (assuming gamma=0.05 for CVaR)
                cvar = compute_cvar(loss, 0.05)
                
                # Plot the mean utility vs CVaR
                ax.scatter(np.mean(utility), cvar, label=f'{folder}', alpha=0.7)
        
        # Customize the plot for this experiment
        ax.set_title(f'{experiment_label} - Mean vs CVaR Tradeoff')
        ax.set_xlabel('Mean Utility')
        ax.set_ylabel('CVaR (gamma=0.05)')
        ax.legend()

        # Save the plot as an image in the "figures" folder
        plot_filename = f"figures/2_Mean_vs_CVaR_tradeoff_{experiment_label}.png"
        fig.savefig(plot_filename)
        plt.close(fig)  # Close the figure after saving to avoid display

    print("All plots have been saved in the 'figures' folder.")


def plot_cvar_vs_w(results_folder):
    """
    Draw the plot of CVaR vs Measurement Budget (W) for different methods.
    :param results_folder: Path to the folder where results of the experiments are saved.
    """
    # Define the experiments and their respective labels
    experiments = [
        {"label": "Gamma_r", "folders": ["exp1_gamma_r=0.6", "exp1_gamma_r=1.0", "exp1_gamma_r=1.5", "exp1_gamma_r=2.0"], "gamma": None},
        {"label": "Lambda", "folders": ["exp2_lambda_true=0.5", "exp2_lambda_true=1.5", "exp2_lambda_true=5.0"], "gamma": None},
        {"label": "W (Measurement Budget)", "folders": ["exp3_W=100", "exp3_W=20", "exp3_W=500"], "gamma": None},
        {"label": "L (Sample Size)", "folders": ["exp4_L=128", "exp4_L=256", "exp4_L=64"], "gamma": None},
    ]
    
    # Store data for the CVaR vs W plot
    cvar_vs_w_data = []

    for experiment in experiments:
        for folder in experiment["folders"]:
            folder_path = os.path.join(results_folder, folder, "checkpoints")
            file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('TRAIN')]
            
            for file_path in file_paths:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Extract loss and W values
                loss = df['loss_train_mean']
                
                # For "exp3_W" folder, extract the W value from the folder name
                if "exp3_W" in folder:
                    W_value = int(folder.split('=')[-1])  # Extract W value from folder name
                    
                    # Compute CVaR for the losses (assuming gamma=0.05 for CVaR)
                    cvar = compute_cvar(loss, 0.05)
                    
                    # Collect data for CVaR vs W plot
                    cvar_vs_w_data.append({"method": folder, "W": W_value, "CVaR": cvar})

    # Plot CVaR vs W
    cvar_vs_w_df = pd.DataFrame(cvar_vs_w_data)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=cvar_vs_w_df, x='W', y='CVaR', hue='method', marker="o")
    plt.title("CVaR vs Measurement Budget (W)")
    plt.xlabel("Measurement Budget (W)")
    plt.ylabel("CVaR (gamma=0.05)")
    plt.legend()
    
    # Save the plot to the "figures" folder
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    plt.savefig("figures/3_CVaR_vs_W.png")
    plt.close()  # Close the plot after saving
    print("CVaR vs Measurement Budget (W) plot has been saved in the 'figures' folder.")

def plot_loss_distribution_sorted(results_folder):
    """
    Draw separate Boxplots / Violin plots of loss distribution for the 4 experiments (Gamma_r, Lambda, W, L),
    sorted by their respective values (L, W, Lambda).
    Save each plot separately with unique filenames.
    
    :param results_folder: Path to the folder where results of the experiments are saved.
    """
    # Define the experiments and their respective labels
    experiments = [
        {"label": "Gamma_r", "folders": ["exp1_gamma_r=0.6", "exp1_gamma_r=1.0", "exp1_gamma_r=1.5", "exp1_gamma_r=2.0"]},
        {"label": "Lambda", "folders": ["exp2_lambda_true=0.5", "exp2_lambda_true=1.5", "exp2_lambda_true=5.0"]},
        {"label": "W (Measurement Budget)", "folders": ["exp3_W=100", "exp3_W=20", "exp3_W=500"]},
        {"label": "L (Sample Size)", "folders": ["exp4_L=128", "exp4_L=256", "exp4_L=64"]},
    ]
    
    # Ensure the "figures" folder exists
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    # Process each experiment
    for experiment in experiments:
        loss_distribution_data = []

        for folder in experiment["folders"]:
            folder_path = os.path.join(results_folder, folder, "checkpoints")
            file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('TRAIN')]
            
            for file_path in file_paths:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Extract loss data and method for the plot
                loss = df['loss_train_mean']
                method = folder.split('=')[-1]  # Extract method name (e.g., "gamma_r=0.6")

                for l in loss:
                    loss_distribution_data.append({"method": method, "loss": l})
        
        # Convert data to DataFrame
        loss_distribution_df = pd.DataFrame(loss_distribution_data)
        
        # Sort the methods based on the relevant values (Lambda, L, W)
        if experiment['label'] == "Gamma_r":
            sorted_methods = sorted(loss_distribution_df['method'].unique(), key=lambda x: float(x.split('=')[-1]))
        elif experiment['label'] == "Lambda":
            sorted_methods = sorted(loss_distribution_df['method'].unique(), key=lambda x: float(x.split('=')[-1]))
        elif experiment['label'] == "W (Measurement Budget)":
            sorted_methods = sorted(loss_distribution_df['method'].unique(), key=lambda x: int(x.split('=')[-1]))
        elif experiment['label'] == "L (Sample Size)":
            sorted_methods = sorted(loss_distribution_df['method'].unique(), key=lambda x: int(x.split('=')[-1]))

        # Sort DataFrame based on the sorted methods
        loss_distribution_df['method'] = pd.Categorical(loss_distribution_df['method'], categories=sorted_methods, ordered=True)
        loss_distribution_df = loss_distribution_df.sort_values('method')
        
        # Plot the Boxplots / Violin of Loss Distribution
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='method', y='loss', data=loss_distribution_df, palette="Set2")
        plt.title(f"Boxplots / Violin of Loss Distribution for {experiment['label']}")
        plt.xlabel("Method")
        plt.ylabel("Loss")
        
        # Save the plot to the "figures" folder with unique filenames
        plot_filename = f"figures/4_Loss_Distribution_Sorted_{experiment['label']}.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the plot after saving
        
        print(f"Loss Distribution plot for {experiment['label']} has been saved as {plot_filename}.")


def compute_zipf_exponent(popularity_values):
    """
    Estimate the Zipf exponent based on popularity values using a logarithmic fit.
    :param popularity_values: List or array of popularity values.
    :return: Estimated Zipf exponent.
    """
    ranks = np.arange(1, len(popularity_values) + 1)
    log_ranks = np.log(ranks)
    log_popularity = np.log(popularity_values)
    # Fit a line to log(ranks) vs log(popularity)
    slope, intercept = np.polyfit(log_ranks, log_popularity, 1)
    return -slope  # The exponent is the negative slope

def plot_performance_vs_zipf(results_folder, gamma_values=[0.6, 1.0, 1.5, 2.0]): # heat map 
    """
    Draw Performance vs Zipf Exponent plot for CVaR and Mean Utility for each gamma value.
    :param results_folder: Path to the folder where results of the experiments are saved.
    :param gamma_values: List of gamma values to use for experiments (default: [0.6, 1.0, 1.5, 2.0]).
    """
    # Define the experiment folders for exp1 (gamma_r)
    experiment_folders = [f"exp1_gamma_r={gamma}" for gamma in gamma_values]

    # Store results for plotting
    zipf_exponent_data = []

    for folder in experiment_folders:
        folder_path = os.path.join(results_folder, folder, "checkpoints")
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('TRAIN')]

        for file_path in file_paths:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract loss and utility data
            loss = df['loss_train_mean']
            utility = df['mean_utility_mean_RL2O-CVaR']
            
            # Compute CVaR for the losses (assuming gamma=0.05 for CVaR)
            cvar = compute_cvar(loss, 0.05)
            
            # Compute Zipf exponent from the utility distribution
            zipf_exponent = compute_zipf_exponent(utility)
            
            # Collect data for the heatmap
            zipf_exponent_data.append({"method": folder, "zipf_exponent": zipf_exponent, "cvar": cvar, "mean_utility": np.mean(utility)})

    # Convert data to DataFrame
    zipf_exponent_df = pd.DataFrame(zipf_exponent_data)

    # Plotting the CVaR vs Zipf Exponent
    plt.figure(figsize=(10, 7))

    # Create heatmap for CVaR
    pivot_cvar = zipf_exponent_df.pivot(index="method", columns="zipf_exponent", values="cvar")
    sns.heatmap(pivot_cvar, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)
    plt.title(f"CVaR vs Zipf Exponent for gamma_r values")
    plt.xlabel("Zipf Exponent")
    plt.ylabel("Method")
    plt.tight_layout()
    
    # Save the CVaR heatmap
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    plt.savefig("figures/6_Performance_vs_Zipf_CVaR.png")
    plt.close()  # Close the plot after saving

    # Plotting the Mean Utility vs Zipf Exponent
    plt.figure(figsize=(10, 7))

    # Create heatmap for Mean Utility
    pivot_utility = zipf_exponent_df.pivot(index="method", columns="zipf_exponent", values="mean_utility")
    sns.heatmap(pivot_utility, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title(f"Mean Utility vs Zipf Exponent for gamma_r values")
    plt.xlabel("Zipf Exponent")
    plt.ylabel("Method")
    plt.tight_layout()
    
    # Save the Mean Utility heatmap
    plt.savefig("figures/6_Performance_vs_Zipf_Utility.png")
    plt.close()  # Close the plot after saving

    print("CVaR and Mean Utility vs Zipf Exponent heatmaps have been saved in the 'figures' folder.")


def plot_cvar_vs_l(results_folder):
    """
    Draw CVaR vs L (Sample Size) plot for exp4 (L = {64, 128, 256}).
    :param results_folder: Path to the folder where results of the experiments are saved.
    """
    # Define the experiment folders for exp4 (L = {64, 128, 256})
    experiment_folders = ["exp4_L=64", "exp4_L=128", "exp4_L=256"]

    # Store results for plotting
    cvar_data = []

    for folder in experiment_folders:
        folder_path = os.path.join(results_folder, folder, "checkpoints")
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('TRAIN')]

        for file_path in file_paths:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract CVaR_0.05_* data (assuming column names like "CVaR_0.05_RL2O-CVaR")
            cvar_05 = df['CVaR_0.05_RL2O-CVaR'].values[0]  # Take the first value for each file (or adjust accordingly)

            # Collect data for plotting
            sample_size = int(folder.split('=')[1])  # Extract L value from folder name
            cvar_data.append({"L": sample_size, "cvar": cvar_05})

    # Convert data to DataFrame
    cvar_df = pd.DataFrame(cvar_data)

    # Plotting CVaR vs L (Sample Size)
    plt.figure(figsize=(8, 6))
    plt.plot(cvar_df['L'], cvar_df['cvar'], marker='o', linestyle='-', color='b')
    plt.title("CVaR vs Sample Size (L)")
    plt.xlabel("Sample Size (L)")
    plt.ylabel("CVaR (gamma=0.05)")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    plt.savefig("figures/8_CVaR_vs_L.png")
    plt.close()  # Close the plot after saving

    print("CVaR vs L plot has been saved in the 'figures' folder.")

def plot_training_curves(results_folder='results', smoothing_window=10):
    """
    Plots training and validation CVaR curves for each experiment.
    - Smoothed empirical CVaR objective vs. training epoch.
    - Validation CVaR vs. epoch for different methods.
    :param results_folder: Path to the folder where experiment results are saved.
    :param smoothing_window: The window size for the rolling average of the training loss.
    """
    # Ensure the "figures" directory exists
    os.makedirs("figures", exist_ok=True)

    method_linestyles = {
        "RL2O-CVaR": "--",
        "Plug-in Mean-Opt": ":",
        "Popularity Heuristic (Top-S)": "-."
    }
    
    # Define a color palette for the parameter values
    color_palette = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']

    for experiment in experiments:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Assign colors to each parameter value in the experiment
        param_colors = {folder.split('=')[-1]: color_palette[i % len(color_palette)] for i, folder in enumerate(experiment["folders"])}

        experiment_label = experiment["label"]
        
        for folder in experiment["folders"]:
            folder_path = os.path.join(results_folder, folder, "checkpoints")
            # Find the training data file
            file_paths = glob.glob(os.path.join(folder_path, 'TRAIN*.csv'))
            
            if not file_paths:
                print(f"No training CSV found in {folder_path}")
                continue

            # Assuming one training file per folder
            df = pd.read_csv(file_paths[0])
            param_value = folder.split('=')[-1]
            color = param_colors[param_value]
            
            # --- Plot 1: Smoothed Training Loss (CVaR Objective) ---
            smoothed_loss = df['loss_train_mean'].rolling(window=smoothing_window, min_periods=1).mean()
            ax1.plot(df['epoch'], smoothed_loss, '-', color=color, label=f'Train Loss ({experiment["label"]}={param_value})', alpha=0.8)

            # --- Plot 2: Validation CVaR ---
            for method, linestyle in method_linestyles.items():
                cvar_col = f'CVaR_0.05_{method}'
                
                if cvar_col in df.columns:
                    ax1.plot(df['epoch'], df[cvar_col], linestyle=linestyle, color=color, label=f'Val CVaR ({method}, {param_value})')

        # --- Formatting the plot ---
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('CVaR / Smoothed Loss')
        fig.suptitle(f'Training and Validation CVaR Curves: {experiment_label}', fontsize=16)
        
        # Create a legend
        ax1.legend(loc='upper right')
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

        # Save the plot
        plot_filename = f"figures/9_Training_Curves_{experiment_label}.png"
        plt.savefig(plot_filename)
        plt.close(fig)

    print("All training curve plots have been saved in the 'figures' folder.")

plot_training_curves()
# def generate_plots():
    # Utility_CDFs('results')
    # Mean_vs_CVaR_tradeoff_scatter('results')
#     CVaR_Va_vs_measurement_budget_W()

# plot_cvar_vs_w('results')
# plot_loss_distribution_sorted('results')
# plot_performance_vs_zipf('results')
# plot_cvar_vs_l('results')