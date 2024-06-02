import matplotlib.pyplot as plt

# Define the data
data = [18, 11, 30, 8, 5, 102, 67, 248, 18, 131, 16, 291, 43, 147, 128, 99, 52, 80, 260, 100, 158, 53, 137, 118, 96, 24, 85, 329, 83, 106, 342, 23, 37, 182, 33, 40, 111, 81, 112, 116, 239, 24, 24, 5, 2, 22, 36, 56]

# Create a plot of the total 1s
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(data) + 1), data, color='lightcoral')
plt.xlabel('Timestep')
plt.ylabel('Number of Positive Classes')
plt.title('Total Positive Classes for each Timestep')
plt.show()