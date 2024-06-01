import pandas as pd
import matplotlib.pyplot as plt
import csv

# Initialize an empty list to store the data from each round
data = []

# Extract the relevant data from each round
for i in range(1, 5):
    file_path = f"round{i}_hyperparameter_tuning_results.csv"
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            params = eval(row[0])
            metrics = eval(row[1])
            f1_score = metrics[12][1]
            data.append({'round': i, 'params': params, 'f1_score': f1_score})

# Create a DataFrame from the combined data
df = pd.DataFrame(data)

# Plot the F1 score results
plt.figure(figsize=(12, 6))
plt.scatter(range(len(df)), df['f1_score'], color='blue')
plt.plot(range(len(df)), df['f1_score'], color='blue')

plt.xlabel('Hyperparameter Configuration')
plt.ylabel('F1 Score')
plt.title('F1 Score Results Across All Rounds')
plt.show()