import ast
import pandas as pd
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

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'NPV', 'FPR', 'FDR', 'MCC', 'ROC_AUC', 'AUCPR']
file_paths = {
    'LSTM': '/Users/ariasarch/JAWZ_Big_Data/LSTM_ASXGB_elliptic_AF_results.csv',
    'Simple': '/Users/ariasarch/JAWZ_Big_Data/Simple_LSTM_ASXGB_elliptic_AF_results.csv',
    'ASXGB': '/Users/ariasarch/JAWZ_Big_Data/ASXGB_elliptic_AF_results.csv',
    'AXGBp': '/Users/ariasarch/JAWZ_Big_Data/AXGBp_elliptic_AF_results.csv',
    'AXGBr': '/Users/ariasarch/JAWZ_Big_Data/AXGBr_elliptic_AF_results.csv',
    'ARF': '/Users/ariasarch/JAWZ_Big_Data/ARF_elliptic_AF_results.csv'
}

data_dict = {model_name: load_and_process_metrics(path) for model_name, path in file_paths.items()}
# plot_metrics(data_dict, metric_names)
highest_means_count = count_highest_means(data_dict, metric_names)

print("Number of times each dataset had the highest mean for metrics:")
for dataset, count in highest_means_count.items():
    print(f"{dataset}: {count}")

def plot_metrics_over_time(data_dict, metric_names):
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        for model_name, data in data_dict.items():
            if model_name == 'LSTM':
                if metric in data.columns:
                    plt.plot(data['timestep'], data[metric], label=model_name, marker='o')
        plt.title(f"{metric} Over Time")
        plt.xlabel('Timestep')
        plt.ylabel(f'{metric} Value')
        plt.legend()
        plt.grid(True)
        plt.show()

plot_metrics_over_time(data_dict, metric_names)
