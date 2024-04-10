import csv
import matplotlib.pyplot as plt
import numpy as np

from functions import load_csv


#load data
#data has [sepal_length, sepal_width. petal_length, petal_width]
class_1 = load_csv('Iris_TTT4275/class_1', False)
class_2 = load_csv('Iris_TTT4275/class_2', False)
class_3 = load_csv('Iris_TTT4275/class_3', False)

# Get the number of dimensions
num_dimensions = len(class_1[0])

# Create subplots for each dimension
fig, axs = plt.subplots(num_dimensions, 1, figsize=(10, 8), sharex=True)

# Plot histograms for each dimension of each class
for dim in range(num_dimensions):
    axs[dim].hist([class_1[i][dim] for i in range(len(class_1))], alpha=0.5, label='Class 1', color='red')
    axs[dim].hist([class_2[i][dim] for i in range(len(class_2))], alpha=0.5, label='Class 2', color='green')
    axs[dim].hist([class_3[i][dim] for i in range(len(class_3))], alpha=0.5, label='Class 3', color='blue')
    axs[dim].set_title(f'Histogram for Dimension {dim+1}')
    axs[dim].set_ylabel('Amount')

# Add common x-axis label
axs[num_dimensions - 1].set_xlabel('Value')

# Add legend
axs[0].legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()