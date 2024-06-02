import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Function to load each csv
def load_and_process_metrics(file_path):
    
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Parse the 'metrics' column
    data['metrics'] = data['metrics'].apply(ast.literal_eval)
    
    # Initialize columns for each metric
    for metric in metric_names:
        data[metric] = data['metrics'].apply(lambda metrics: dict(metrics).get(metric, None))
    
    return data
    
def plot_metrics(data_dict, metric_names):
    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, data in data_dict.items():
            avg = data[metric].mean()
            se = data[metric].sem()
            ax.bar(model_name, avg, yerr=se, label=model_name, capsize=5)
        ax.set_title(f"{metric} Comparison")
        ax.set_ylabel('Metric Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

def count_highest_means(data_dict, metric_names):
    highest_count = {model_name: 0 for model_name in data_dict.keys()}
    for metric in metric_names:
        max_mean = max((1 - data[metric].mean() if metric in ['FPR', 'FDR'] else data[metric].mean(), model_name) for model_name, data in data_dict.items())
        highest_count[max_mean[1]] += 1
    return highest_count

def plot_metrics_over_time(data_dict, metric_names, title, clarifier):
    # Determine the number of subplots needed
    num_models = len(data_dict)
    
    # Create a 2x2 grid for subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes into a 1D array for easy iteration
    
    for ax, (model_name, data) in zip(axes, data_dict.items()):
        for metric in metric_names:
            if metric == 'F1 Score' and metric in data.columns and 'timestep' in data.columns:
                ax.plot(data['timestep'], data[metric], label=model_name, marker='o')
                ax.set_title(f"{model_name} ({title} - {clarifier}): {metric}")
                ax.set_xlabel('Timestep')
                ax.set_ylabel(f'{metric} Value')
                ax.legend()
    
    # Hide any unused subplots (if less than 4 models)
    for ax in axes[num_models:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_average_f1_score(data_dict, title, clarifier):
    model_names = []
    average_f1_scores = []
    std_f1_scores = []
    for model_name, data in data_dict.items():
        if 'F1 Score' in data.columns:
            model_names.append(model_name)
            f1_scores = data['F1 Score']
            average_f1_score = f1_scores.mean()
            std_f1_score = f1_scores.std()
            average_f1_scores.append(average_f1_score)
            std_f1_scores.append(std_f1_score)
    # Define a list of colors for each model
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, average_f1_scores, yerr=std_f1_scores, capsize=5, color=colors[:len(model_names)])
    plt.title(f"({title} - {clarifier}) Average F1 Score")
    plt.xlabel("Model")
    plt.ylabel("Average F1 Score")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'NPV', 'FPR', 'FDR', 'MCC', 'ROC_AUC', 'AUCPR']

lf_file_paths = {
    'LSTM_ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_ASXGB_elliptic_LF_results.csv',
    'LSTM': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_Base_elliptic_LF_results.csv',
    'ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/ASXGB_elliptic_LF_results.csv',
    'ARF': '/Users/ariasarch/JAWZ_Big_Data/Results/ARF_elliptic_LF_results.csv'
}

af_file_paths = {
    'LSTM_ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_ASXGB_elliptic_AF_results.csv',
    'LSTM': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_Base_elliptic_AF_results.csv',
    'ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/ASXGB_elliptic_AF_results.csv',
    'ARF': '/Users/ariasarch/JAWZ_Big_Data/Results/ARF_elliptic_AF_results.csv'
}

lf_norolling_file_paths = {
    'LSTM_ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_ASXGB_elliptic_LF_results_norolling.csv',
    'LSTM': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_Base_elliptic_LF_results_norolling.csv',
    'ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/ASXGB_elliptic_LF_results_norolling.csv',
    'ARF': '/Users/ariasarch/JAWZ_Big_Data/Results/ARF_elliptic_LF_results_norolling.csv'
}

af_norolling_file_paths = {
    'LSTM_ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_ASXGB_elliptic_AF_results_norolling.csv',
    'LSTM': '/Users/ariasarch/JAWZ_Big_Data/Results/LSTM_Base_elliptic_AF_results_norolling.csv',
    'ASXGB': '/Users/ariasarch/JAWZ_Big_Data/Results/ASXGB_elliptic_AF_results_norolling.csv',
    'ARF': '/Users/ariasarch/JAWZ_Big_Data/Results/ARF_elliptic_AF_results_norolling.csv'
}

file_paths = [lf_file_paths, af_file_paths, lf_norolling_file_paths, af_norolling_file_paths]

for i, file_path_dict in enumerate(file_paths):
    if i == 0 or i == 2:
        title = 'LF'
    else:
        title = 'AF'

    if i == 2 or i == 3:
        clarifier = 'Entire Timeseries'
    else:
        clarifier = 'Rolling Window'

    data_dict = {model_name: load_and_process_metrics(path) for model_name, path in file_path_dict.items()}
    highest_means_count = count_highest_means(data_dict, metric_names)

    print("Number of times each dataset had the highest mean for metrics:")
    for dataset, count in highest_means_count.items():
        print(f"{dataset}: {count}")

    plot_metrics_over_time(data_dict, metric_names, title, clarifier)
    plot_average_f1_score(data_dict, title, clarifier)

