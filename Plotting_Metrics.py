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

# Function to count which model has the highest metrics 
def count_highest_means(LSTM, ASXGB, AXGBp, AXGBr, ARF, metric_names):
    # Initialize a dictionary to store the count of highest means for each dataset
    highest_count = {
        'LSTM': 0,
        'ASXGB': 0,
        'AXGBp': 0,
        'AXGBr': 0,
        'ARF': 0
    }

    # Iterate over each metric
    for metric in metric_names:
        # Calculate means, inversing if necessary due to the nature of the metric
        means_LSTM = 1 - LSTM[metric].mean() if metric in ['FPR', 'FDR'] else LSTM[metric].mean()
        means_ASXGB = 1 - ASXGB[metric].mean() if metric in ['FPR', 'FDR'] else ASXGB[metric].mean()
        means_AXGBp = 1 - AXGBp[metric].mean() if metric in ['FPR', 'FDR'] else AXGBp[metric].mean()
        means_AXGBr = 1 - AXGBr[metric].mean() if metric in ['FPR', 'FDR'] else AXGBr[metric].mean()
        means_ARF = 1 - ARF[metric].mean() if metric in ['FPR', 'FDR'] else ARF[metric].mean()

        # List to hold the means for easier comparison
        all_means = [means_LSTM, means_ASXGB, means_AXGBp, means_AXGBr, means_ARF]

        # Find the index of the dataset with the highest mean
        highest_index = all_means.index(max(all_means))

        # Map the index to the corresponding dataset name
        if highest_index == 0:
            highest_count['LSTM'] += 1
        elif highest_index == 1:
            highest_count['ASXGB'] += 1
        elif highest_index == 2:
            highest_count['AXGBp'] += 1
        elif highest_index == 3:
            highest_count['AXGBr'] += 1
        elif highest_index == 4:
            highest_count['ARF'] += 1

    return highest_count

# Initialize key variables 
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'NPV', 'FPR', 'FDR', 'MCC', 'ROC_AUC', 'AUCPR']

# Load each csv
data_LSTM = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/LSTM_ASXGB_elliptic_AF_results.csv')
data_ASXGB = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/ASXGB_elliptic_AF_results.csv')
data_AXGBp = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/AXGBp_elliptic_AF_results.csv')
data_AXGBr = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/AXGBr_elliptic_AF_results.csv')
data_ARF = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/ARF_elliptic_AF_results.csv')

# Plotting each metric as a separate figure
for metric in metric_names:
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust figure size as needed

    # Calculate average and standard error of metrics for each model
    avg_LSTM = data_LSTM[metric].mean()
    se_LSTM = data_LSTM[metric].sem()
    avg_ASXGB = data_ASXGB[metric].mean()
    se_ASXGB = data_ASXGB[metric].sem()
    avg_AXGBp = data_AXGBp[metric].mean()
    se_AXGBp = data_AXGBp[metric].sem()
    avg_AXGBr = data_AXGBr[metric].mean()
    se_AXGBr = data_AXGBr[metric].sem()
    avg_ARF = data_ARF[metric].mean()
    se_ARF = data_ARF[metric].sem()

    # Bar plot with error bars
    ax.bar('LSTM', avg_LSTM, yerr=se_LSTM, color='black', width=0.4, capsize=5, label='LSTM')
    ax.bar('ASXGB', avg_ASXGB, yerr=se_ASXGB, color='skyblue', width=0.4, capsize=5, label='ASXGB')
    ax.bar('AXGBp', avg_AXGBp, yerr=se_AXGBp, color='grey', width=0.4, capsize=5, label='AXGBp')
    ax.bar('AXGBr', avg_AXGBr, yerr=se_AXGBr, color='lightgreen', width=0.4, capsize=5, label='AXGBr')
    ax.bar('ARF', avg_ARF, yerr=se_ARF, color='salmon', width=0.4, capsize=5, label='ARF')

    ax.set_title(f"{metric} Comparison")
    ax.set_ylabel('Metric Value')

    plt.tight_layout()
    plt.show()

plt.tight_layout()
plt.show()

# Call the function and store the result
highest_means_count = count_highest_means(data_LSTM, data_ASXGB, data_AXGBp, data_AXGBr, data_ARF, metric_names)

# Print the result
print("Number of times each dataset had the highest mean for metrics:")
for dataset, count in highest_means_count.items():
    print(f"{dataset}: {count}")