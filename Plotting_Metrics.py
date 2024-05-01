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

# Function to check and report significant differences
def check_significance(data1, data2, data3, metric_names):
    significant_results = {}
    for metric in metric_names:
        all_data = pd.concat([
            pd.Series(data1[metric], name=metric),
            pd.Series(data2[metric], name=metric),
            pd.Series(data3[metric], name=metric)
        ], axis=0).dropna()
        groups = ['ASXGB'] * len(data1) + ['AXGBr'] * len(data2) + ['ARF'] * len(data3)
        
        f_val, p_val = stats.f_oneway(data1[metric].dropna(), data2[metric].dropna(), data3[metric].dropna())
        
        if p_val < 0.05:
            tukey = pairwise_tukeyhsd(endog=all_data, groups=groups, alpha=0.05)
            significant_results[metric] = tukey.summary()
    
    return significant_results

# Function to count which model has the highest metrics 
def count_highest_means(ASXGB, AXGBp, AXGBr, ARF, metric_names):
    # Initialize a dictionary to store the count of highest means for each dataset
    highest_count = {
        'ASXGB': 0,
        'AXGBp': 0,
        'AXGBr': 0,
        'ARF': 0
    }

    # Iterate over each metric
    for metric in metric_names:
        # Calculate means, inversing if necessary due to the nature of the metric
        means_ASXGB = 1 - ASXGB[metric].mean() if metric in ['FPR', 'FDR'] else ASXGB[metric].mean()
        means_AXGBp = 1 - AXGBp[metric].mean() if metric in ['FPR', 'FDR'] else AXGBp[metric].mean()
        means_AXGBr = 1 - AXGBr[metric].mean() if metric in ['FPR', 'FDR'] else AXGBr[metric].mean()
        means_ARF = 1 - ARF[metric].mean() if metric in ['FPR', 'FDR'] else ARF[metric].mean()

        # List to hold the means for easier comparison
        all_means = [means_ASXGB, means_AXGBp, means_AXGBr, means_ARF]

        # Find the index of the dataset with the highest mean
        highest_index = all_means.index(max(all_means))

        # Map the index to the corresponding dataset name
        if highest_index == 0:
            highest_count['ASXGB'] += 1
        elif highest_index == 1:
            highest_count['AXGBp'] += 1
        elif highest_index == 2:
            highest_count['AXGBr'] += 1
        elif highest_index == 3:
            highest_count['ARF'] += 1

    return highest_count

# Initialize key variables 
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'NPV', 'FPR', 'FDR', 'MCC', 'ROC_AUC', 'AUCPR']

# Load each csv
data_ASXGB = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/ASXGB_elliptic_AF_results.csv')
data_AXGBp = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/AXGBp_elliptic_AF_results.csv')
data_AXGBr = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/AXGBr_elliptic_AF_results.csv')
data_ARF = load_and_process_metrics('/Users/ariasarch/JAWZ_Big_Data/ARF_elliptic_AF_results.csv')

# Plotting each metric as a bar graph, averaging over all timesteps
fig, axs = plt.subplots(len(metric_names), 1, figsize=(10, 5 * len(metric_names)))  # Adjust subplot size as needed
for i, metric in enumerate(metric_names):
    # Calculate average and standard error of metrics for each model
    avg_ASXGB = data_ASXGB[metric].mean()
    se_ASXGB = data_ASXGB[metric].sem()
    avg_AXGBp = data_AXGBp[metric].mean()
    se_AXGBp = data_AXGBp[metric].sem()
    avg_AXGBr = data_AXGBr[metric].mean()
    se_AXGBr = data_AXGBr[metric].sem()
    avg_ARF = data_ARF[metric].mean()
    se_ARF = data_ARF[metric].sem()

    # Bar plot with error bars
    axs[i].bar('ASXGB', avg_ASXGB, yerr=se_ASXGB, color='skyblue', width=0.4, capsize=5)
    axs[i].bar('AXGBp', avg_AXGBp, yerr=se_AXGBp, color='grey', width=0.4, capsize=5)
    axs[i].bar('AXGBr', avg_AXGBr, yerr=se_AXGBr, color='lightgreen', width=0.4, capsize=5)
    axs[i].bar('ARF', avg_ARF, yerr=se_ARF, color='salmon', width=0.4, capsize=5)

    # # Scatter plot overlay for each model
    # axs[i].scatter(['ASXGB'] * len(data_ASXGB), data_ASXGB[metric], color='darkblue', alpha=0.7)
    # axs[i].scatter(['AXGBr'] * len(data_AXGBr), data_AXGBr[metric], color='darkgreen', alpha=0.7)
    # axs[i].scatter(['ARF'] * len(data_ARF), data_ARF[metric], color='darkred', alpha=0.7)

    axs[i].set_title(metric)
    axs[i].set_ylabel('Metric Value')

plt.tight_layout()
plt.show()

# Check and print significant results
# results = check_significance(data_ASXGB, data_AXGBr, data_ARF, metric_names)
# for metric, result in results.items():
#     print(f"Significant Results for {metric}:\n{result}\n")

# Call the function and store the result
highest_means_count = count_highest_means(data_ASXGB, data_AXGBp, data_AXGBr, data_ARF, metric_names)

# Print the result
print("Number of times each dataset had the highest mean for metrics:")
for dataset, count in highest_means_count.items():
    print(f"{dataset}: {count}")