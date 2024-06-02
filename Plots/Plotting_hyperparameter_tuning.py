import matplotlib.pyplot as plt

# Data for plotting
categories = ["Units", "Epochs", "Dropout", "Learning rate", "Weight Decay", "Scale Factor"]

hyperparameters = [
    [32, 64, 128],
    [5, 10, 20, 40],
    [0.2, 0.3, 0.4, 0.5],
    [0.0001, 0.00001, 0.000001],
    [0.1, 0.01, 0.001],
    [0.9, 1.0, 1.1]
]

f1_scores = [
    [0.30, 0.25, 0.15],
    [0.11, 0.25, 0.33, 0.33],
    [0.33, 0.24, 0.33, 0.24],
    [0.21, 0.14, 0.30],
    [0.35, 0.34, 0.33],
    [0.30, 0.29, 0.27]
]


# Define colors from lightest to darkest green
colors = [
    '#d0f0c0', # lightest green
    '#a8e6a3', # light green
    '#70d47b', # moderate green
    '#48c35d', # dark moderate green
    '#28b240', # dark green
    '#0a9f24'  # darkest green
]

# Define the positions for each bar group
positions = [0, 1, 2, 3, 4, 5]

# Corrected plotting with each category having a distinct shade of green and separated bars
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each category with a different shade of green
current_position = 0
for i, (cat, scores, color) in enumerate(zip(categories, f1_scores, colors)):
    x_positions = [current_position + j for j in range(len(scores))]
    ax.bar(x_positions, scores, label=cat, color=color)
    current_position += len(scores) + 1

# Adding dashed vertical lines to separate sections
current_position = 0
for i, scores in enumerate(f1_scores[:-1]):
    current_position += len(scores)
    plt.axvline(x=current_position + i, color='gray', linestyle='--')

# Adding headers for each section
tick_positions = []
current_position = 0
for scores in f1_scores:
    tick_positions.append(current_position + len(scores) / 2 - 0.5)
    current_position += len(scores) + 1

plt.xticks(tick_positions, categories)

plt.xlabel("Hyperparameters")
plt.ylabel("F1 Scores")
plt.title("Individual Tuning Results")

plt.show()

